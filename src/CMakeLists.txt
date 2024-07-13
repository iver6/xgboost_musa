file(GLOB_RECURSE CPU_SOURCES *.cc *.h)
list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/cli_main.cc)

#-- Object library
# Object library is necessary for jvm-package, which creates its own shared library.
add_library(objxgboost OBJECT)
target_sources(objxgboost PRIVATE ${CPU_SOURCES})
# Skip files with factory object
set_source_files_properties(
  predictor/predictor.cc gbm/gbm.cc tree/tree_updater.cc metric/metric.cc objective/objective.cc
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES *.cu *.cuh)
  target_sources(objxgboost PRIVATE ${CUDA_SOURCES})
endif()

if(USE_MUSA)

  list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake) 
  
  find_package(MUSA REQUIRED)

  file(GLOB_RECURSE MUSA_SOURCES *.mu *.muh)

  set(MUSA_VERBOSE_BUILD ON)
  set(MUSA_MCC_FLAGS --offload-arch=mp_21 -Werror -D__MUSACC__ -std=c++17 -DXGBOOST_USE_MUSA=1)
  musa_include_directories(${xgboost_SOURCE_DIR}/include)
  musa_include_directories(${xgboost_SOURCE_DIR}/dmlc-core/include)

  musa_add_library(musa_obj STATIC ${MUSA_SOURCES})

  #musa_compile(MUSA_OBJ ${MUSA_SOURCES})
  #target_link_libraries(objxgboost PRIVATE ${MUSA_OBJ})
  #target_sources(objxgboost PRIVATE ${MUSA_SOURCES})
  #message(STATUS "MUSA_OBJ is ${MUSA_OBJ}")
endif()

if(PLUGIN_SYCL)
  target_compile_definitions(objxgboost PRIVATE -DXGBOOST_USE_SYCL=1)
endif()

target_include_directories(objxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include)

if(LOG_CAPI_INVOCATION)
  target_compile_definitions(objxgboost PRIVATE -DLOG_CAPI_INVOCATION=1)
endif()

# This grouping organises source files nicely in visual studio
auto_source_group("${CUDA_SOURCES}")
auto_source_group("${CPU_SOURCES}")
#auto_source_group("${MUSA_SOURCES}")

#-- End object library
