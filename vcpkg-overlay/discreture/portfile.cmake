vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO rustyconover/discreture
    REF eafe006f906e9fa6f79eec0ddd76eeb7de1676e0
    SHA512 ca293ac3ce7ce9fbcabbbad3e52eceb67dc224ec69e5f2b92b749a4740dab80cb7c1fa5e53eb51be86ecdf95414849e84cbbeea80bfa23d81010a61468b58aa3
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
