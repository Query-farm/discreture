#include "discreture_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <optional>
#include "discreture.hpp"

namespace duckdb {

struct PartitionsBindData : public TableFunctionData {
	int n;
	PartitionsBindData(int n_p) : n(n_p) {
	}
};

struct CombBindData : public TableFunctionData {
	int n;
	int k;
	CombBindData(int n_p, int k_p) : n(n_p), k(k_p) {
	}
};

struct PermBindData : public TableFunctionData {
	int n;
	PermBindData(int n_p) : n(n_p) {
	}
};

// ----------- Global State Structures -----------

struct CombGlobalState : public GlobalTableFunctionState {
public:
	const discreture::Combinations<int> comb;

	std::mutex mutex;
	const idx_t k;

private:
	std::vector<discreture::Combinations<int>::iterator> work;
	idx_t current_work_idx;

public:
	CombGlobalState(int n, int k_val, int threads)
	    : comb(n, k_val), work(discreture::divide_work_in_equal_parts(comb.begin(), comb.end(), threads)), k(k_val),
	      current_work_idx(0) {
	}

	std::optional<idx_t> GetWorkIdx() {
		std::lock_guard<std::mutex> lock(mutex);
		auto result = current_work_idx++;
		if (result > work.size() - 2) {
			return std::nullopt;
		}
		return result;
	}

	discreture::Combinations<int>::iterator &GetIter(idx_t idx) {
		return work[idx];
	}

	inline bool IsLastIndex(idx_t idx) {
		return idx == work.size() - 2;
	}
};

struct CombLocalState : public LocalTableFunctionState {
	std::optional<idx_t> work_idx;

	CombLocalState() {
		work_idx = std::nullopt;
	}
};

struct PermGlobalState : public GlobalTableFunctionState {
	discreture::Permutations<int> perm;
	discreture::Permutations<int>::iterator it;
	idx_t n;

	PermGlobalState(int n_val) : perm(n_val), it(perm.begin()), n(n_val) {
	}
};

// ----------- Bind Functions -----------

static unique_ptr<FunctionData> CombBind(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	int n = input.inputs[0].GetValue<int32_t>();
	int k = input.inputs[1].GetValue<int32_t>();

	if (n <= 0 || k <= 0 || k > n) {
		throw BinderException("Invalid arguments for combinations: require n > 0, 0 < k <= n");
	}

	for (int i = 0; i < k; i++) {
		return_types.push_back(LogicalType::INTEGER);
		names.push_back("col" + std::to_string(i));
	}

	return make_uniq<CombBindData>(n, k);
}

static unique_ptr<FunctionData> PermBind(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	int n = input.inputs[0].GetValue<int32_t>();

	if (n <= 0) {
		throw BinderException("Invalid argument for permutations: require n > 0");
	}

	for (int i = 0; i < n; i++) {
		return_types.push_back(LogicalType::INTEGER);
		names.push_back("col" + std::to_string(i));
	}

	return make_uniq<PermBindData>(n);
}

// ----------- Init Global State -----------

static unique_ptr<GlobalTableFunctionState> CombInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<CombBindData>();
	return make_uniq<CombGlobalState>(bind_data.n, bind_data.k, context.db->config.options.maximum_threads);
}

unique_ptr<LocalTableFunctionState> CombInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                  GlobalTableFunctionState *global_state) {
	//	auto &comb_global_state = global_state->Cast<CombGlobalState>();
	return make_uniq<CombLocalState>();
}

static unique_ptr<GlobalTableFunctionState> PermInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<PermBindData>();
	return make_uniq<PermGlobalState>(bind_data.n);
}

// ----------- Execution Functions -----------

static void CombExec(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &global_state = data.global_state->Cast<CombGlobalState>();
	auto &local_state = data.local_state->Cast<CombLocalState>();

	auto output_vectors = output.data;

	// I want to get these pointers for all contents of the output columns
	auto data_ptrs = vector<int32_t *>(output.ColumnCount());
	data_ptrs.reserve(output.ColumnCount());
	for (idx_t col = 0; col < output.ColumnCount(); col++) {
		data_ptrs[col] = FlatVector::GetData<int32_t>(output_vectors[col]);
	}

	idx_t count = 0;
	while (count < STANDARD_VECTOR_SIZE) {
		if (!local_state.work_idx.has_value()) {
			// Lock the mutex in the global state
			local_state.work_idx = global_state.GetWorkIdx();
			if (!local_state.work_idx.has_value()) {
				output.SetCardinality(count);
				return;
			}
		}

		bool is_last = global_state.IsLastIndex(*local_state.work_idx);

		auto &iterator = global_state.GetIter(*local_state.work_idx);
		auto &end = global_state.GetIter(*local_state.work_idx + 1);

		if (is_last) {
			while (count < STANDARD_VECTOR_SIZE && iterator != global_state.comb.end()) {
				const auto &comb = *iterator;
				for (idx_t col = 0; col < global_state.k; col++) {
					data_ptrs[col][count] = comb[col];
				}
				++iterator;
				count++;
			}
			if (iterator == global_state.comb.end()) {
				local_state.work_idx = std::nullopt;
			}
		} else {
			while (count < STANDARD_VECTOR_SIZE && iterator != end) {
				const auto &comb = *iterator;
				for (idx_t col = 0; col < global_state.k; col++) {
					data_ptrs[col][count] = comb[col];
				}
				++iterator;
				count++;
			}
			if (iterator == end) {
				local_state.work_idx = std::nullopt;
			}
		}
		output.SetCardinality(count);
	}
}

static void PermExec(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<PermGlobalState>();
	idx_t count = 0;

	auto output_vectors = output.data;

	// I want to get these pointers for all contents of the output columns
	auto data_ptrs = vector<int32_t *>(state.n);
	for (idx_t col = 0; col < state.n; col++) {
		data_ptrs[col] = FlatVector::GetData<int32_t>(output_vectors[col]);
	}

	while (state.it != state.perm.end() && count < STANDARD_VECTOR_SIZE) {
		for (idx_t col = 0; col < state.n; col++) {
			data_ptrs[col][count] = (*state.it)[col];
		}
		++state.it;
		count++;
	}
	output.SetCardinality(count);
}

unique_ptr<NodeStatistics> CombCardinality(ClientContext &context, const FunctionData *bind_data) {
	auto &comb_bind_data = bind_data->Cast<CombBindData>();

	discreture::Combinations<int> comb(comb_bind_data.n, comb_bind_data.k);
	const auto distance = std::distance(comb.begin(), comb.end());

	return make_uniq<NodeStatistics>(distance, distance);
}

unique_ptr<NodeStatistics> PermCardinality(ClientContext &context, const FunctionData *bind_data) {
	auto &perm_bind_data = bind_data->Cast<PermBindData>();

	discreture::Permutations<int> perm(perm_bind_data.n);
	const auto distance = std::distance(perm.begin(), perm.end());

	return make_uniq<NodeStatistics>(distance, distance);
}

unique_ptr<BaseStatistics> CombStatistics(ClientContext &context, const FunctionData *bind_data,
                                          column_t column_index) {
	auto &comb_bind_data = bind_data->Cast<CombBindData>();

	auto r = NumericStats::CreateEmpty(LogicalType::INTEGER);
	NumericStats::SetMin(r, 0);
	NumericStats::SetMax(r, comb_bind_data.n - 1);
	r.SetHasNoNull();
	r.SetDistinctCount(comb_bind_data.n);
	return r.ToUnique();
}

unique_ptr<BaseStatistics> PermStatistics(ClientContext &context, const FunctionData *bind_data,
                                          column_t column_index) {
	auto &perm_bind_data = bind_data->Cast<PermBindData>();

	auto r = NumericStats::CreateEmpty(LogicalType::INTEGER);
	NumericStats::SetMin(r, 0);
	NumericStats::SetMax(r, perm_bind_data.n - 1);
	r.SetHasNoNull();
	r.SetDistinctCount(perm_bind_data.n);
	return r.ToUnique();
}

static void LoadInternal(DatabaseInstance &instance) {
	TableFunction combinations_func("combinations", {LogicalType::INTEGER, LogicalType::INTEGER}, CombExec, CombBind,
	                                CombInitGlobal, CombInitLocal);

	combinations_func.cardinality = CombCardinality;
	combinations_func.statistics = CombStatistics;

	TableFunction permutations_func("permutations", {LogicalType::INTEGER}, PermExec, PermBind, PermInitGlobal);
	permutations_func.cardinality = PermCardinality;
	permutations_func.statistics = PermStatistics;

	ExtensionUtil::RegisterFunction(instance, combinations_func);
	ExtensionUtil::RegisterFunction(instance, permutations_func);
}

void DiscretureExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string DiscretureExtension::Name() {
	return "discreture";
}

std::string DiscretureExtension::Version() const {
#ifdef EXT_VERSION_QUACK
	return EXT_VERSION_QUACK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void discreture_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::DiscretureExtension>();
}

DUCKDB_EXTENSION_API const char *discreture_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}
