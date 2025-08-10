vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO rustyconover/discreture
    REF 26bdd8a84e5637badb18551fabd4582c1715fed6
    SHA512 69d24c5400077c557c501d0f7877b48ad67afb0c16e4c9f24c21b1935903429c5353771a5b56f4b464ba82ba4be20925b28dbb967a71a7f8fcc4fd8b9edded89
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
