add_executable(sine "./sine.cpp")
target_link_libraries(sine ${GOOGLETEST_LIBS})
add_dependencies(sine TGT_qpmad TGT_googletest TGT_cppad)
add_test(NAME sine COMMAND sine)


add_executable(paraboloid "./paraboloid.cpp")
target_link_libraries(paraboloid ${GOOGLETEST_LIBS})
add_dependencies(paraboloid TGT_qpmad TGT_googletest TGT_cppad)
add_test(NAME paraboloid COMMAND paraboloid)


add_executable(generic "./generic.cpp")
target_link_libraries(generic ${GOOGLETEST_LIBS})
add_dependencies(generic TGT_qpmad TGT_googletest TGT_cppad)
add_test(NAME generic COMMAND generic)
