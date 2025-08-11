vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO rustyconover/discreture
    REF 1f35a06a1fc15c965d49bf00626ead8bb2f30999
    SHA512 9f14feed222534b0ee5be2a6608dcc1af9d9be649ce3cc5394f1b7c0fd91bc83f70964fadd5eced07cb9439992c3cea5173c31cf480d2817f90752e8c40dc321
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
