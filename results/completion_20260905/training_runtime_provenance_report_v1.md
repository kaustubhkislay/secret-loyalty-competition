# Original training image provenance recovery

Modal metadata recovered image `im-KEoh7oytKRiTq3pC8HWpLJ` from the original stopped training app `ap-q4lwgh1bEwsVAeFaftw9Ap`.
The app layout contains one unique image. All twelve saved child calls resolve to its `train_pair` function, `fu-iETg5po2jyUZvrcDJPbLRn`.
The archived source assigns one shared image to all four functions. The retrieved call graph reports successful inputs for the parent and twelve children.

The retained image metadata lists 101 Python distributions, Python 3.12.10, glibc 2.36, and image builder version 2025.06.
All 98 Linux-applicable lock pins match. The other two pins, `colorama` and `tzdata`, have markers for other platforms.
The image also contains `pip==25.1.1`, `uv==0.7.19`, and `wheel==0.45.1` from its base environment.
Every saved Python and seven-package version record from the twelve training results matches this image.

A CPU-only probe instantiated that retained image directly by ID. It performed no installation, rebuild, training, or model inference.
The probe had no secrets, mounted volumes, GPU, or outbound network. It returned the same 101 distribution versions and an empty error stream.
Its inventory also records distribution metadata and RECORD-file hashes.
The script saved sandbox handle `sb-DEH4JOlm2eJao0wyzlYIdG` before waiting. The probe app was `ap-euVwpBRwF58JGpgAe3QzIv`.

These results recover the original app image identity and its retained package inventory without reconstructing the lock in a new environment.
They do not recover a contemporaneous inventory from each historical GPU process. Runtime injection, driver versions, and unrecorded process changes remain unknown.
The SDK responses contain no OCI/content build digest. The image ID, builder version, and evidence-file hashes have distinct meanings.

`training_runtime_recovery_v2/` contains the read-only RPC evidence and the exact retrieval script.
`training_runtime_image_probe_v1/` contains the explicit request, pre-wait handles, inventory, empty stderr, and successful outcome.
`training_runtime_recovery_v1/` preserves the initial connection failure.
The accompanying JSON report binds every evidence file by SHA-256 and records each historical version comparison.
