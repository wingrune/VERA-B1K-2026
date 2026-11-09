from importlib import import_module

__all__ = [

    "FeatureType",
    "NormalizationMode",
    "PipelineFeatureType",
    "PolicyFeature",
    "RTCAttentionSchedule",

    "DatasetRecordConfig",
    "DatasetConfig",
    "EvalConfig",
    "PeftConfig",
    "PreTrainedConfig",
    "WandBConfig",
    "VideoEncoderConfig",
    "DepthEncoderConfig",

    "camera_encoder_defaults",
    "depth_encoder_defaults",

    "VALID_VIDEO_CODECS",
    "VIDEO_ENCODER_INFO_KEYS",
]

_LAZY_EXPORTS = {
    "DatasetRecordConfig": ".dataset",
    "DatasetConfig": ".default",
    "EvalConfig": ".default",
    "PeftConfig": ".default",
    "WandBConfig": ".default",
    "PreTrainedConfig": ".policies",
    "FeatureType": ".types",
    "NormalizationMode": ".types",
    "PipelineFeatureType": ".types",
    "PolicyFeature": ".types",
    "RTCAttentionSchedule": ".types",
    "VALID_VIDEO_CODECS": ".video",
    "VIDEO_ENCODER_INFO_KEYS": ".video",
    "DepthEncoderConfig": ".video",
    "VideoEncoderConfig": ".video",
    "camera_encoder_defaults": ".video",
    "depth_encoder_defaults": ".video",
}


def __getattr__(name: str) -> object:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
