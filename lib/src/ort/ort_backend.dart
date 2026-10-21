/// Native ONNX Runtime backend (FFI). Imported only on `dart:ffi` targets.
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as p;

import '../execution_control.dart';
import '../onnx_graph.dart' show ExecutionProfile;
import '../onnx_runner.dart';
import '../tensor.dart';
import 'ort_exception.dart';
import 'ort_options.dart';

typedef _Status = Pointer<Void>;
typedef _Out = Pointer<Pointer<Void>>;

/// ORT C API slots used by this backend. The layout is append-only; these
/// indices are the v17 layout used by ORT 1.15+ and must be validated by the
/// smoke test whenever the pinned API version changes.
abstract final class _Slots {
  static const getErrorMessage = 2;
  static const createEnv = 3;
  static const createSession = 7;
  static const createSessionFromArray = 8;
  static const run = 9;
  static const createSessionOptions = 10;
  static const setSessionExecutionMode = 13;
  static const enableMemPattern = 16;
  static const disableMemPattern = 17;
  static const enableCpuMemArena = 18;
  static const disableCpuMemArena = 19;
  static const setGraphOptimizationLevel = 23;
  static const setIntraOpNumThreads = 24;
  static const setInterOpNumThreads = 25;
  static const sessionGetInputCount = 30;
  static const sessionGetOutputCount = 31;
  static const sessionGetInputTypeInfo = 33;
  static const sessionGetOutputTypeInfo = 34;
  static const sessionGetInputName = 36;
  static const sessionGetOutputName = 37;
  static const createTensorWithData = 49;
  static const getTensorMutableData = 51;
  static const castTypeInfoToTensorInfo = 55;
  static const getTensorElementType = 60;
  static const getDimensionsCount = 61;
  static const getDimensions = 62;
  static const getTensorShapeElementCount = 64;
  static const getTensorTypeAndShape = 65;
  static const createCpuMemoryInfo = 69;
  static const allocatorFree = 76;
  static const getAllocatorWithDefaultOptions = 78;
  static const releaseEnv = 92;
  static const releaseStatus = 93;
  static const releaseMemoryInfo = 94;
  static const releaseSession = 95;
  static const releaseValue = 96;
  static const releaseTypeInfo = 98;
  static const releaseTensorTypeAndShapeInfo = 99;
  static const releaseSessionOptions = 100;
  static const getAvailableProviders = 125;
  static const releaseAvailableProviders = 126;
  static const appendCudaProvider = 204;
  static const createCudaProviderOptions = 205;
  static const updateCudaProviderOptions = 206;
  static const releaseCudaProviderOptions = 208;
}

/// A loaded ONNX Runtime shared library and its C API table.
final class OrtRuntime {
  OrtRuntime._(this.library, this._api, this.version);

  final DynamicLibrary library;
  final Pointer<Pointer<Void>> _api;

  /// The ONNX Runtime version string reported by the library.
  final String version;

  static OrtRuntime load(OrtOptions options) {
    final wantsCuda = options.providers.any((p) => p is OrtCudaProvider);
    final library = _openLibrary(
      options.libraryPath,
      preloadCuda: options.preloadCudaDependencies && wantsCuda,
      cudaDirectory: options.cudaLibraryDirectory,
    );
    final base = library
        .lookupFunction<
          Pointer<Pointer<Void>> Function(),
          Pointer<Pointer<Void>> Function()
        >('OrtGetApiBase')();
    if (base == nullptr) {
      throw const OrtException('OrtGetApiBase returned null');
    }
    final version = base[1]
        .cast<NativeFunction<Pointer<Utf8> Function()>>()
        .asFunction<Pointer<Utf8> Function()>()()
        .toDartString();
    final api = base[0]
        .cast<NativeFunction<Pointer<Pointer<Void>> Function(Uint32)>>()
        .asFunction<Pointer<Pointer<Void>> Function(int)>()(17);
    if (api == nullptr) {
      throw OrtException(
        'ONNX Runtime $version does not expose C API v17; install 1.15+',
      );
    }
    return OrtRuntime._(library, api, version);
  }

  /// Loads a model from [path]. External-data companions are resolved by ORT
  /// relative to the model file.
  OrtNativeSession openFile(String path, OrtOptions options) =>
      OrtNativeSession._openFile(this, path, options);

  /// Loads a self-contained model from [bytes]. Models with external data must
  /// use [openFile].
  OrtNativeSession openBytes(Uint8List bytes, OrtOptions options) =>
      OrtNativeSession._openBytes(this, bytes, options);
}

/// An ONNX Runtime session. Internal; users interact through [OrtRunner].
final class OrtNativeSession {
  OrtNativeSession._(
    this._api,
    this._env,
    this._session,
    this._memory,
    this._inputNames,
    this._outputNames,
    this._inputSpecs,
    this._outputSpecs,
    this._providers,
    this._warnings,
  );

  final Pointer<Pointer<Void>> _api;
  Pointer<Void> _env;
  Pointer<Void> _session;
  Pointer<Void> _memory;
  bool _closed = false;

  final List<String> _inputNames;
  final List<String> _outputNames;
  final List<TensorSpec> _inputSpecs;
  final List<TensorSpec> _outputSpecs;
  final List<String> _providers;
  final List<String> _warnings;

  List<String> get inputNames => _inputNames;
  List<String> get outputNames => _outputNames;
  List<TensorSpec> get inputSpecs => _inputSpecs;
  List<TensorSpec> get outputSpecs => _outputSpecs;
  List<String> get providers => _providers;
  List<String> get warnings => _warnings;

  factory OrtNativeSession._openFile(
    OrtRuntime runtime,
    String path,
    OrtOptions options,
  ) => _create(runtime, path, null, options);

  factory OrtNativeSession._openBytes(
    OrtRuntime runtime,
    Uint8List bytes,
    OrtOptions options,
  ) => _create(runtime, null, bytes, options);

  static OrtNativeSession _create(
    OrtRuntime runtime,
    String? path,
    Uint8List? bytes,
    OrtOptions options,
  ) {
    final api = runtime._api;
    final warnings = <String>[];
    final usedProviders = <String>{'CPUExecutionProvider'};
    Pointer<Void> env = nullptr;
    Pointer<Void> session = nullptr;
    Pointer<Void> memory = nullptr;
    Pointer<Void> sessionOptions = nullptr;
    try {
      using((a) {
        final out = a<Pointer<Void>>();
        _check(
          api,
          api[_Slots.createEnv]
              .cast<
                NativeFunction<_Status Function(Int32, Pointer<Utf8>, _Out)>
              >()
              .asFunction<_Status Function(int, Pointer<Utf8>, _Out)>()(
            3,
            'onnx_runtime_dart'.toNativeUtf8(allocator: a),
            out,
          ),
        );
        env = out.value;
        out.value = nullptr;
        _check(
          api,
          api[_Slots.createSessionOptions]
              .cast<NativeFunction<_Status Function(_Out)>>()
              .asFunction<_Status Function(_Out)>()(out),
        );
        sessionOptions = out.value;
        out.value = nullptr;

        _setInt(
          api,
          _Slots.setGraphOptimizationLevel,
          sessionOptions,
          options.graphOptimization.ortValue,
        );
        _setInt(
          api,
          _Slots.setIntraOpNumThreads,
          sessionOptions,
          options.intraOpThreads,
        );
        _setInt(
          api,
          _Slots.setInterOpNumThreads,
          sessionOptions,
          options.interOpThreads,
        );
        _setInt(
          api,
          _Slots.setSessionExecutionMode,
          sessionOptions,
          options.executionMode.ortValue,
        );
        _toggle(
          api,
          options.enableMemPattern
              ? _Slots.enableMemPattern
              : _Slots.disableMemPattern,
          sessionOptions,
        );
        _toggle(
          api,
          options.enableCpuMemArena
              ? _Slots.enableCpuMemArena
              : _Slots.disableCpuMemArena,
          sessionOptions,
        );

        final providers = options.providers.isEmpty
            ? const <OrtProvider>[OrtCpuProvider()]
            : options.providers;
        for (final provider in providers) {
          switch (provider) {
            case OrtCpuProvider():
              // ONNX Runtime always registers the CPU provider.
              break;
            case OrtCudaProvider():
              final available = _availableProviders(api);
              if (!available.contains('CUDAExecutionProvider')) {
                warnings.add(
                  'CUDAExecutionProvider unavailable; using CPU provider',
                );
                break;
              }
              _appendCuda(api, sessionOptions, provider, a);
              usedProviders.add('CUDAExecutionProvider');
            case OrtCoreMlProvider():
            case OrtDirectMlProvider():
            case OrtNnapiProvider():
              throw UnsupportedError(
                '${provider.runtimeType} is not wired yet; use CPU or CUDA',
              );
          }
        }

        if (path != null) {
          final modelPath = p.absolute(path);
          final nativePath = Platform.isWindows
              ? modelPath.toNativeUtf16(allocator: a).cast<Void>()
              : modelPath.toNativeUtf8(allocator: a).cast<Void>();
          _check(
            api,
            api[_Slots.createSession]
                .cast<
                  NativeFunction<
                    _Status Function(
                      Pointer<Void>,
                      Pointer<Void>,
                      Pointer<Void>,
                      _Out,
                    )
                  >
                >()
                .asFunction<
                  _Status Function(
                    Pointer<Void>,
                    Pointer<Void>,
                    Pointer<Void>,
                    _Out,
                  )
                >()(env, nativePath, sessionOptions, out),
          );
        } else {
          final model = bytes!;
          final nativeBytes = a<Uint8>(math.max(1, model.length));
          nativeBytes.asTypedList(model.length).setAll(0, model);
          _check(
            api,
            api[_Slots.createSessionFromArray]
                .cast<
                  NativeFunction<
                    _Status Function(
                      Pointer<Void>,
                      Pointer<Uint8>,
                      Size,
                      Pointer<Void>,
                      _Out,
                    )
                  >
                >()
                .asFunction<
                  _Status Function(
                    Pointer<Void>,
                    Pointer<Uint8>,
                    int,
                    Pointer<Void>,
                    _Out,
                  )
                >()(env, nativeBytes, model.length, sessionOptions, out),
          );
        }
        session = out.value;
        out.value = nullptr;

        _check(
          api,
          api[_Slots.createCpuMemoryInfo]
              .cast<NativeFunction<_Status Function(Int32, Int32, _Out)>>()
              .asFunction<_Status Function(int, int, _Out)>()(1, 0, out),
        );
        memory = out.value;
        out.value = nullptr;
      });

      final inputNames = <String>[];
      final outputNames = <String>[];
      final inputSpecs = <TensorSpec>[];
      final outputSpecs = <TensorSpec>[];
      using((a) {
        final allocator = a<Pointer<Void>>();
        _check(
          api,
          api[_Slots.getAllocatorWithDefaultOptions]
              .cast<NativeFunction<_Status Function(_Out)>>()
              .asFunction<_Status Function(_Out)>()(allocator),
        );
        final count = a<Size>();
        _check(
          api,
          api[_Slots.sessionGetInputCount]
              .cast<
                NativeFunction<_Status Function(Pointer<Void>, Pointer<Size>)>
              >()
              .asFunction<_Status Function(Pointer<Void>, Pointer<Size>)>()(
            session,
            count,
          ),
        );
        for (var i = 0; i < count.value; i++) {
          final name = _sessionName(
            api,
            _Slots.sessionGetInputName,
            session,
            i,
            allocator.value,
            a,
          );
          inputNames.add(name);
          final info = _modelInfo(
            api,
            _Slots.sessionGetInputTypeInfo,
            session,
            i,
            a,
          );
          inputSpecs.add(TensorSpec(name, info.$2, info.$1));
        }
        _check(
          api,
          api[_Slots.sessionGetOutputCount]
              .cast<
                NativeFunction<_Status Function(Pointer<Void>, Pointer<Size>)>
              >()
              .asFunction<_Status Function(Pointer<Void>, Pointer<Size>)>()(
            session,
            count,
          ),
        );
        for (var i = 0; i < count.value; i++) {
          final name = _sessionName(
            api,
            _Slots.sessionGetOutputName,
            session,
            i,
            allocator.value,
            a,
          );
          outputNames.add(name);
          final info = _modelInfo(
            api,
            _Slots.sessionGetOutputTypeInfo,
            session,
            i,
            a,
          );
          outputSpecs.add(TensorSpec(name, info.$2, info.$1));
        }
      });

      for (final warning in warnings) {
        stderr.writeln('onnx_runtime_dart: $warning');
      }
      return OrtNativeSession._(
        api,
        env,
        session,
        memory,
        inputNames,
        outputNames,
        inputSpecs,
        outputSpecs,
        [
          if (usedProviders.contains('CUDAExecutionProvider'))
            'CUDAExecutionProvider',
          'CPUExecutionProvider',
        ],
        List.unmodifiable(warnings),
      );
    } catch (_) {
      if (session != nullptr) _release(api, _Slots.releaseSession, session);
      if (sessionOptions != nullptr) {
        _release(api, _Slots.releaseSessionOptions, sessionOptions);
      }
      if (memory != nullptr) _release(api, _Slots.releaseMemoryInfo, memory);
      if (env != nullptr) _release(api, _Slots.releaseEnv, env);
      rethrow;
    }
  }

  void _validateInputs(Map<String, Tensor> inputs) {
    final names = _inputNames.toSet();
    for (final key in inputs.keys) {
      if (!names.contains(key)) {
        throw ArgumentError('Unknown input "$key"; expected $inputNames');
      }
    }
    for (final spec in _inputSpecs) {
      final t = inputs[spec.name];
      if (t == null) {
        throw ArgumentError('Missing input "${spec.name}"');
      }
      if (t.rank != spec.shape.length) {
        throw ArgumentError(
          'Input "${spec.name}" rank ${t.rank} != '
          'model rank ${spec.shape.length}',
        );
      }
      for (var i = 0; i < spec.shape.length; i++) {
        final want = spec.shape[i];
        if (want > 0 && want != t.shape[i]) {
          throw ArgumentError(
            'Input "${spec.name}" dim $i is ${t.shape[i]}, '
            'model declares $want',
          );
        }
      }
    }
  }

  Map<String, Tensor> run(
    Map<String, Tensor> inputs,
    List<String> requestedOutputs, {
    ExecutionProfile? profile,
    OnnxExecutionControl? control,
  }) {
    if (_closed) throw const OrtException('Session is closed');
    if (requestedOutputs.isEmpty) return {};
    for (final name in requestedOutputs) {
      if (!_outputNames.contains(name)) {
        throw ArgumentError('Unknown output "$name"; expected $_outputNames');
      }
    }
    _validateInputs(inputs);
    control?.checkpoint();

    final arena = Arena();
    final values = <Pointer<Void>>[];
    final outputs = <Pointer<Void>>[];
    try {
      final inputNamesPtr = arena<Pointer<Utf8>>(_inputSpecs.length);
      final inputValuesPtr = arena<Pointer<Void>>(_inputSpecs.length);
      for (var i = 0; i < _inputSpecs.length; i++) {
        final spec = _inputSpecs[i];
        final tensor = inputs[spec.name]!;
        inputNamesPtr[i] = spec.name.toNativeUtf8(allocator: arena);
        final native = _writeTensor(tensor, spec.elemType, arena);
        final dims = arena<Int64>(tensor.rank);
        dims.asTypedList(tensor.rank).setAll(0, tensor.shape);
        final valueOut = arena<Pointer<Void>>();
        _check(
          _api,
          _api[_Slots.createTensorWithData]
              .cast<
                NativeFunction<
                  _Status Function(
                    Pointer<Void>,
                    Pointer<Void>,
                    Size,
                    Pointer<Int64>,
                    Size,
                    Int32,
                    _Out,
                  )
                >
              >()
              .asFunction<
                _Status Function(
                  Pointer<Void>,
                  Pointer<Void>,
                  int,
                  Pointer<Int64>,
                  int,
                  int,
                  _Out,
                )
              >()(
            _memory,
            native.$1,
            native.$2,
            dims,
            tensor.rank,
            native.$3,
            valueOut,
          ),
        );
        values.add(valueOut.value);
        inputValuesPtr[i] = valueOut.value;
      }

      final outNamesPtr = arena<Pointer<Utf8>>(requestedOutputs.length);
      final outValuesPtr = arena<Pointer<Void>>(requestedOutputs.length);
      for (var i = 0; i < requestedOutputs.length; i++) {
        outNamesPtr[i] = requestedOutputs[i].toNativeUtf8(allocator: arena);
      }
      _check(
        _api,
        _api[_Slots.run]
            .cast<
              NativeFunction<
                _Status Function(
                  Pointer<Void>,
                  Pointer<Void>,
                  Pointer<Pointer<Utf8>>,
                  Pointer<Pointer<Void>>,
                  Size,
                  Pointer<Pointer<Utf8>>,
                  Size,
                  Pointer<Pointer<Void>>,
                )
              >
            >()
            .asFunction<
              _Status Function(
                Pointer<Void>,
                Pointer<Void>,
                Pointer<Pointer<Utf8>>,
                Pointer<Pointer<Void>>,
                int,
                Pointer<Pointer<Utf8>>,
                int,
                Pointer<Pointer<Void>>,
              )
            >()(
          _session,
          nullptr,
          inputNamesPtr,
          inputValuesPtr,
          _inputSpecs.length,
          outNamesPtr,
          requestedOutputs.length,
          outValuesPtr,
        ),
      );
      for (var i = 0; i < requestedOutputs.length; i++) {
        outputs.add(outValuesPtr[i]);
      }

      final result = <String, Tensor>{};
      for (var i = 0; i < outputs.length; i++) {
        result[requestedOutputs[i]] = _readOutput(outputs[i], arena);
      }
      control?.checkpoint();
      return result;
    } finally {
      for (final value in outputs) {
        _release(_api, _Slots.releaseValue, value);
      }
      for (final value in values) {
        _release(_api, _Slots.releaseValue, value);
      }
      arena.releaseAll();
    }
  }

  Tensor _readOutput(Pointer<Void> value, Arena arena) {
    final infoOut = arena<Pointer<Void>>();
    _check(
      _api,
      _api[_Slots.getTensorTypeAndShape]
          .cast<NativeFunction<_Status Function(Pointer<Void>, _Out)>>()
          .asFunction<_Status Function(Pointer<Void>, _Out)>()(value, infoOut),
    );
    final info = infoOut.value;
    try {
      final meta = _readInfo(_api, info, arena);
      final count = arena<Size>();
      _check(
        _api,
        _api[_Slots.getTensorShapeElementCount]
            .cast<
              NativeFunction<_Status Function(Pointer<Void>, Pointer<Size>)>
            >()
            .asFunction<_Status Function(Pointer<Void>, Pointer<Size>)>()(
          info,
          count,
        ),
      );
      final dataOut = arena<Pointer<Void>>();
      _check(
        _api,
        _api[_Slots.getTensorMutableData]
            .cast<NativeFunction<_Status Function(Pointer<Void>, _Out)>>()
            .asFunction<_Status Function(Pointer<Void>, _Out)>()(
          value,
          dataOut,
        ),
      );
      return _readTensor(meta.$1, meta.$2, dataOut.value, count.value);
    } finally {
      _release(_api, _Slots.releaseTensorTypeAndShapeInfo, info);
    }
  }

  void close() {
    if (_closed) return;
    _closed = true;
    _release(_api, _Slots.releaseSession, _session);
    _release(_api, _Slots.releaseMemoryInfo, _memory);
    _release(_api, _Slots.releaseEnv, _env);
    _session = nullptr;
    _memory = nullptr;
    _env = nullptr;
  }
}

/// User-facing FFI-backed runner.
final class OrtRunner implements OnnxRunner {
  OrtRunner._(this._session, this.runtimeVersion);

  final OrtNativeSession _session;

  /// Version string of the loaded ONNX Runtime library.
  final String runtimeVersion;

  /// Loads a model synchronously. The FFI session setup itself is synchronous;
  /// prefer [fromFile] in async code.
  static OrtRunner fromFileSync(
    String path, {
    OrtOptions options = const OrtOptions(),
  }) {
    final runtime = OrtRuntime.load(options);
    final session = runtime.openFile(path, options);
    return OrtRunner._(session, runtime.version);
  }

  /// Loads a self-contained model synchronously.
  static OrtRunner fromBytesSync(
    Uint8List bytes, {
    OrtOptions options = const OrtOptions(),
  }) {
    final runtime = OrtRuntime.load(options);
    final session = runtime.openBytes(bytes, options);
    return OrtRunner._(session, runtime.version);
  }

  static Future<OrtRunner> fromFile(
    String path, {
    OrtOptions options = const OrtOptions(),
  }) async => fromFileSync(path, options: options);

  static Future<OrtRunner> fromBytes(
    Uint8List bytes, {
    OrtOptions options = const OrtOptions(),
  }) async => fromBytesSync(bytes, options: options);

  @override
  List<TensorSpec> get inputSpecs => _session.inputSpecs;

  @override
  List<String> get outputNames => _session.outputNames;

  List<TensorSpec> get outputSpecs => _session.outputSpecs;

  /// Providers available to this session, in priority order. This reflects
  /// what was actually registered; a requested CUDA provider that is not built
  /// into the loaded library shows up as CPU only.
  List<String> get providers => _session.providers;

  /// Non-fatal warnings emitted while creating the session (for example, a
  /// requested CUDA provider that is not available in the loaded library).
  List<String> get warnings => _session.warnings;

  @override
  Map<String, Tensor> run(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
    OnnxExecutionControl? control,
  }) => _session.run(inputs, outputNames, profile: profile, control: control);

  /// ONNX Runtime's C call is synchronous; this schedules the call on the
  /// calling isolate's event loop. Use a worker isolate for UI work.
  @override
  Future<Map<String, Tensor>> runAsync(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
    OnnxExecutionControl? control,
  }) => Future(
    () => run(inputs, outputNames, profile: profile, control: control),
  );

  @override
  void dispose() => _session.close();
}

// ---------------------------------------------------------------------------
// ORT C API helpers
// ---------------------------------------------------------------------------

void _setInt(
  Pointer<Pointer<Void>> api,
  int slot,
  Pointer<Void> target,
  int value,
) {
  _check(
    api,
    api[slot]
        .cast<NativeFunction<_Status Function(Pointer<Void>, Int32)>>()
        .asFunction<_Status Function(Pointer<Void>, int)>()(target, value),
  );
}

void _toggle(Pointer<Pointer<Void>> api, int slot, Pointer<Void> target) {
  _check(
    api,
    api[slot]
        .cast<NativeFunction<_Status Function(Pointer<Void>)>>()
        .asFunction<_Status Function(Pointer<Void>)>()(target),
  );
}

void _check(Pointer<Pointer<Void>> api, Pointer<Void> status) {
  if (status == nullptr) return;
  final message = api[_Slots.getErrorMessage]
      .cast<NativeFunction<Pointer<Utf8> Function(Pointer<Void>)>>()
      .asFunction<Pointer<Utf8> Function(Pointer<Void>)>()(status)
      .toDartString();
  _release(api, _Slots.releaseStatus, status);
  throw OrtException(message);
}

void _release(Pointer<Pointer<Void>> api, int slot, Pointer<Void> value) {
  if (value == nullptr) return;
  api[slot]
      .cast<NativeFunction<Void Function(Pointer<Void>)>>()
      .asFunction<void Function(Pointer<Void>)>()(value);
}

String _sessionName(
  Pointer<Pointer<Void>> api,
  int slot,
  Pointer<Void> session,
  int index,
  Pointer<Void> allocator,
  Arena arena,
) {
  final out = arena<Pointer<Utf8>>();
  _check(
    api,
    api[slot]
        .cast<
          NativeFunction<
            _Status Function(
              Pointer<Void>,
              Size,
              Pointer<Void>,
              Pointer<Pointer<Utf8>>,
            )
          >
        >()
        .asFunction<
          _Status Function(
            Pointer<Void>,
            int,
            Pointer<Void>,
            Pointer<Pointer<Utf8>>,
          )
        >()(session, index, allocator, out),
  );
  try {
    return out.value.toDartString();
  } finally {
    _check(
      api,
      api[_Slots.allocatorFree]
          .cast<
            NativeFunction<_Status Function(Pointer<Void>, Pointer<Void>)>
          >()
          .asFunction<_Status Function(Pointer<Void>, Pointer<Void>)>()(
        allocator,
        out.value.cast(),
      ),
    );
  }
}

(int, List<int>) _modelInfo(
  Pointer<Pointer<Void>> api,
  int slot,
  Pointer<Void> session,
  int index,
  Arena arena,
) {
  final typeOut = arena<Pointer<Void>>();
  _check(
    api,
    api[slot]
        .cast<NativeFunction<_Status Function(Pointer<Void>, Size, _Out)>>()
        .asFunction<_Status Function(Pointer<Void>, int, _Out)>()(
      session,
      index,
      typeOut,
    ),
  );
  final tensorOut = arena<Pointer<Void>>();
  final type = typeOut.value;
  try {
    _check(
      api,
      api[_Slots.castTypeInfoToTensorInfo]
          .cast<NativeFunction<_Status Function(Pointer<Void>, _Out)>>()
          .asFunction<_Status Function(Pointer<Void>, _Out)>()(type, tensorOut),
    );
    if (tensorOut.value == nullptr) {
      throw const OrtException('Model input/output is not a tensor');
    }
    return _readInfo(api, tensorOut.value, arena);
  } finally {
    _release(api, _Slots.releaseTypeInfo, type);
  }
}

(int, List<int>) _readInfo(
  Pointer<Pointer<Void>> api,
  Pointer<Void> info,
  Arena arena,
) {
  final elemType = arena<Int32>();
  _check(
    api,
    api[_Slots.getTensorElementType]
        .cast<NativeFunction<_Status Function(Pointer<Void>, Pointer<Int32>)>>()
        .asFunction<_Status Function(Pointer<Void>, Pointer<Int32>)>()(
      info,
      elemType,
    ),
  );
  final rank = arena<Size>();
  _check(
    api,
    api[_Slots.getDimensionsCount]
        .cast<NativeFunction<_Status Function(Pointer<Void>, Pointer<Size>)>>()
        .asFunction<_Status Function(Pointer<Void>, Pointer<Size>)>()(
      info,
      rank,
    ),
  );
  final dims = arena<Int64>(math.max(1, rank.value));
  if (rank.value > 0) {
    _check(
      api,
      api[_Slots.getDimensions]
          .cast<
            NativeFunction<
              _Status Function(Pointer<Void>, Pointer<Int64>, Size)
            >
          >()
          .asFunction<_Status Function(Pointer<Void>, Pointer<Int64>, int)>()(
        info,
        dims,
        rank.value,
      ),
    );
  }
  return (elemType.value, dims.asTypedList(rank.value).toList());
}

List<String> _availableProviders(Pointer<Pointer<Void>> api) {
  try {
    return using((a) {
      final out = a<Pointer<Pointer<Utf8>>>();
      final count = a<Int32>();
      _check(
        api,
        api[_Slots.getAvailableProviders]
            .cast<
              NativeFunction<
                _Status Function(
                  Pointer<Pointer<Pointer<Utf8>>>,
                  Pointer<Int32>,
                )
              >
            >()
            .asFunction<
              _Status Function(Pointer<Pointer<Pointer<Utf8>>>, Pointer<Int32>)
            >()(out, count),
      );
      try {
        return [
          for (var i = 0; i < count.value; i++) out.value[i].toDartString(),
        ];
      } finally {
        _check(
          api,
          api[_Slots.releaseAvailableProviders]
              .cast<
                NativeFunction<_Status Function(Pointer<Pointer<Utf8>>, Int32)>
              >()
              .asFunction<_Status Function(Pointer<Pointer<Utf8>>, int)>()(
            out.value,
            count.value,
          ),
        );
      }
    });
  } catch (_) {
    return const [];
  }
}

void _appendCuda(
  Pointer<Pointer<Void>> api,
  Pointer<Void> sessionOptions,
  OrtCudaProvider provider,
  Arena arena,
) {
  final out = arena<Pointer<Void>>();
  _check(
    api,
    api[_Slots.createCudaProviderOptions]
        .cast<NativeFunction<_Status Function(_Out)>>()
        .asFunction<_Status Function(_Out)>()(out),
  );
  final cudaOptions = out.value;
  try {
    final values = <String, String>{
      'device_id': '${provider.deviceId}',
      'use_tf32': provider.useTf32 ? '1' : '0',
      'arena_extend_strategy': provider.arenaExtendStrategy.ortValue,
      'cudnn_conv_algo_search': provider.cudnnConvAlgoSearch.ortValue,
      if (provider.gpuMemLimit != null)
        'gpu_mem_limit': '${provider.gpuMemLimit}',
    };
    final keys = arena<Pointer<Utf8>>(values.length);
    final vals = arena<Pointer<Utf8>>(values.length);
    var i = 0;
    for (final entry in values.entries) {
      keys[i] = entry.key.toNativeUtf8(allocator: arena);
      vals[i] = entry.value.toNativeUtf8(allocator: arena);
      i++;
    }
    _check(
      api,
      api[_Slots.updateCudaProviderOptions]
          .cast<
            NativeFunction<
              _Status Function(
                Pointer<Void>,
                Pointer<Pointer<Utf8>>,
                Pointer<Pointer<Utf8>>,
                Size,
              )
            >
          >()
          .asFunction<
            _Status Function(
              Pointer<Void>,
              Pointer<Pointer<Utf8>>,
              Pointer<Pointer<Utf8>>,
              int,
            )
          >()(cudaOptions, keys, vals, values.length),
    );
    _check(
      api,
      api[_Slots.appendCudaProvider]
          .cast<
            NativeFunction<_Status Function(Pointer<Void>, Pointer<Void>)>
          >()
          .asFunction<_Status Function(Pointer<Void>, Pointer<Void>)>()(
        sessionOptions,
        cudaOptions,
      ),
    );
  } finally {
    _release(api, _Slots.releaseCudaProviderOptions, cudaOptions);
  }
}

// ---------------------------------------------------------------------------
// Tensor marshalling
// ---------------------------------------------------------------------------

(Pointer<Void>, int, int) _writeTensor(
  Tensor tensor,
  int modelType,
  Arena arena,
) {
  final n = tensor.length;
  final count = math.max(1, n);
  switch (modelType) {
    case 1:
      final ptr = arena<Float>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        out[i] = tensor.isFloat ? tensor.f![i] : tensor.intData[i].toDouble();
      }
      return (ptr.cast(), n * 4, 1);
    case 7:
      final ptr = arena<Int64>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        out[i] = tensor.isFloat ? tensor.getI(i) : tensor.intData[i];
      }
      return (ptr.cast(), n * 8, 7);
    case 2:
      final ptr = arena<Uint8>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        out[i] = tensor.isFloat
            ? tensor.getI(i) & 0xFF
            : tensor.intData[i] & 0xFF;
      }
      return (ptr.cast(), n, 2);
    case 3:
      final ptr = arena<Int8>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        out[i] = tensor.isFloat ? tensor.getI(i) : tensor.intData[i];
      }
      return (ptr.cast(), n, 3);
    case 6:
      final ptr = arena<Int32>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        out[i] = tensor.isFloat ? tensor.getI(i) : tensor.intData[i];
      }
      return (ptr.cast(), n * 4, 6);
    case 9:
      final ptr = arena<Uint8>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        final v = tensor.isFloat ? tensor.f![i].toInt() : tensor.intData[i];
        out[i] = v == 0 ? 0 : 1;
      }
      return (ptr.cast(), n, 9);
    case 10:
      final ptr = arena<Uint16>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        final v = tensor.isFloat ? tensor.f![i] : tensor.intData[i].toDouble();
        out[i] = _floatToHalf(v);
      }
      return (ptr.cast(), n * 2, 10);
    case 11:
      final ptr = arena<Double>(count);
      final out = ptr.asTypedList(n);
      for (var i = 0; i < n; i++) {
        out[i] = tensor.isFloat ? tensor.f![i] : tensor.intData[i].toDouble();
      }
      return (ptr.cast(), n * 8, 11);
    default:
      throw UnsupportedError(
        'ONNX Runtime input element type $modelType is not supported',
      );
  }
}

Tensor _readTensor(
  int elemType,
  List<int> shape,
  Pointer<Void> data,
  int count,
) {
  switch (elemType) {
    case 1:
      return Tensor.float(
        Float32List.fromList(data.cast<Float>().asTypedList(count)),
        shape,
      );
    case 7:
      return Tensor.int64(
        Int64List.fromList(data.cast<Int64>().asTypedList(count)),
        shape,
      );
    case 2:
      return Tensor.uint8(
        Uint8List.fromList(data.cast<Uint8>().asTypedList(count)),
        shape,
      );
    case 3:
      return Tensor.int8(
        Int8List.fromList(data.cast<Int8>().asTypedList(count)),
        shape,
      );
    case 6:
      final src = data.cast<Int32>().asTypedList(count);
      return Tensor.int64(Int64List.fromList(src), shape);
    case 9:
      final src = data.cast<Uint8>().asTypedList(count);
      return Tensor.int64(
        Int64List.fromList([for (final v in src) v == 0 ? 0 : 1]),
        shape,
      );
    case 10:
      final src = data.cast<Uint16>().asTypedList(count);
      return Tensor.float(
        Float32List.fromList([for (final v in src) _halfToDouble(v)]),
        shape,
      );
    case 11:
      final src = data.cast<Double>().asTypedList(count);
      return Tensor.float(Float32List.fromList(src), shape);
    case 12:
      final src = data.cast<Uint32>().asTypedList(count);
      return Tensor.int64(Int64List.fromList(src), shape);
    case 13:
      final src = data.cast<Uint64>().asTypedList(count);
      return Tensor.int64(Int64List.fromList(src), shape);
    case 16:
      final src = data.cast<Uint16>().asTypedList(count);
      final bits = Uint32List(count);
      for (var i = 0; i < count; i++) {
        bits[i] = src[i] << 16;
      }
      return Tensor.float(
        Float32List.fromList(bits.buffer.asFloat32List()),
        shape,
      );
    default:
      throw UnsupportedError(
        'ONNX Runtime output element type $elemType is not supported',
      );
  }
}

double _halfToDouble(int h) {
  final sign = (h & 0x8000) != 0 ? -1.0 : 1.0;
  final exp = (h >> 10) & 0x1F;
  final mant = h & 0x3FF;
  if (exp == 0) {
    return sign * mant * math.pow(2, -24);
  }
  if (exp == 31) {
    return mant == 0 ? sign * double.infinity : double.nan;
  }
  return sign * (1 + mant / 1024) * math.pow(2, exp - 15);
}

int _floatToHalf(double value) {
  final f = Float32List(1)..[0] = value;
  final bits = f.buffer.asUint32List()[0];
  final sign = (bits >> 16) & 0x8000;
  final exp = (bits >> 23) & 0xFF;
  final mant = bits & 0x7FFFFF;
  if (exp == 255) {
    return sign | 0x7C00 | (mant != 0 ? 0x200 : 0);
  }
  final e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7C00;
  if (e <= 0) {
    if (e < -10) return sign;
    final m = mant | 0x800000;
    final shift = 14 - e;
    var half = m >> shift;
    if (((m >> (shift - 1)) & 1) != 0) half++;
    return sign | half;
  }
  var half = sign | (e << 10) | (mant >> 13);
  final rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0)) half++;
  return half;
}

// ---------------------------------------------------------------------------
// Library loading and CUDA preload
// ---------------------------------------------------------------------------

DynamicLibrary _openLibrary(
  String? explicitPath, {
  required bool preloadCuda,
  String? cudaDirectory,
}) {
  final candidates = <String>[];
  if (explicitPath != null && explicitPath.isNotEmpty) {
    final absolute = p.absolute(explicitPath);
    if (!File(absolute).existsSync()) {
      throw OrtLibraryNotFoundException(
        'ONNX Runtime library not found: $absolute',
      );
    }
    candidates.add(absolute);
  }
  for (final key in const ['ONNX_RUNTIME_DART_ORT_LIBRARY', 'ORT_LIBRARY']) {
    final value = Platform.environment[key];
    if (value != null && value.trim().isNotEmpty) {
      candidates.add(p.absolute(value.trim()));
    }
  }
  final names = Platform.isWindows
      ? const ['onnxruntime.dll']
      : Platform.isMacOS
      ? const ['libonnxruntime.dylib', 'onnxruntime.dylib']
      : const ['libonnxruntime.so', 'libonnxruntime.so.1'];
  final executableDir = p.dirname(Platform.resolvedExecutable);
  for (final name in names) {
    candidates.add(name);
    candidates.add(p.join(executableDir, name));
  }
  Object? lastError;
  for (final candidate in candidates) {
    try {
      final library = DynamicLibrary.open(candidate);
      if (preloadCuda) {
        try {
          final directory =
              cudaDirectory ??
              _resolveCudaLibraryDirectory(candidate, Platform.environment);
          if (directory != null) _preloadCudaLibraries(directory);
        } catch (_) {
          // The CPU provider remains available; let session creation report
          // provider-specific failures.
        }
      }
      return library;
    } catch (e) {
      lastError = e;
    }
  }
  throw OrtLibraryNotFoundException(
    'Could not load ONNX Runtime. Pass OrtOptions(libraryPath: ...) or set '
    'ONNX_RUNTIME_DART_ORT_LIBRARY. Tried: ${candidates.join(', ')}',
    lastError,
  );
}

String? _resolveCudaLibraryDirectory(
  String library,
  Map<String, String> environment,
) {
  final override =
      environment['ONNX_RUNTIME_DART_CUDA_DIR'] ??
      environment['WEBPT_CUDA_DLL_DIR'];
  if (override != null && override.trim().isNotEmpty) {
    return p.normalize(p.absolute(override.trim()));
  }
  final libraryDirectory = p.dirname(p.absolute(library));
  final executableDirectory = p.dirname(Platform.resolvedExecutable);
  final candidates = <String>{
    libraryDirectory,
    p.join(libraryDirectory, 'cuda12'),
    p.join(p.dirname(libraryDirectory), 'cuda12'),
    p.join(p.dirname(p.dirname(libraryDirectory)), 'cuda12'),
    p.join(executableDirectory, 'cuda12'),
  };
  for (final candidate in candidates) {
    final directory = p.normalize(candidate);
    if (File(p.join(directory, 'cublasLt64_12.dll')).existsSync()) {
      return directory;
    }
  }
  return null;
}

final Set<String> _loadedCudaDirectories = {};
final List<DynamicLibrary> _cudaLibraries = [];

void _preloadCudaLibraries(String directory) {
  if (!Platform.isWindows) return;
  final absolute = p.normalize(p.absolute(directory));
  if (!Directory(absolute).existsSync()) {
    throw FileSystemException('CUDA DLL directory does not exist', absolute);
  }
  if (_loadedCudaDirectories.contains(absolute)) return;
  final kernel = DynamicLibrary.open('kernel32.dll');
  final addDirectory = kernel
      .lookupFunction<
        Pointer<Void> Function(Pointer<Utf16>),
        Pointer<Void> Function(Pointer<Utf16>)
      >('AddDllDirectory');
  final defaults = kernel
      .lookupFunction<Int32 Function(Uint32), int Function(int)>(
        'SetDefaultDllDirectories',
      );
  using((a) {
    if (defaults(0x1000) == 0) {
      throw StateError('Cannot enable Windows CUDA DLL search directories.');
    }
    final cookie = addDirectory(absolute.toNativeUtf16(allocator: a));
    if (cookie == nullptr) {
      throw StateError('Cannot register CUDA DLL directory: $absolute');
    }
  });
  for (final name in const [
    'cudart64_12.dll',
    'cublasLt64_12.dll',
    'cublas64_12.dll',
    'cudnn64_9.dll',
  ]) {
    final path = p.join(absolute, name);
    if (File(path).existsSync()) _cudaLibraries.add(DynamicLibrary.open(path));
  }
  _loadedCudaDirectories.add(absolute);
}
