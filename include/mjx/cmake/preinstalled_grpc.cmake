set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Threads REQUIRED)
# Prefer Protobuf's CMake CONFIG package, but fall back to the module-mode
# finder (FindProtobuf) for distributions (e.g. Ubuntu) that ship libprotobuf
# without a ProtobufConfig.cmake. Both provide the protobuf::libprotobuf target.
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf MODULE REQUIRED)
endif()
find_package(gRPC CONFIG REQUIRED)

message(STATUS "Using Protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
find_program(_PROTOBUF_PROTOC protoc)
set(_GRPC_GRPCPP gRPC::grpc++)
set(_REFLECTION gRPC::grpc++_reflection)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
