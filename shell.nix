{
  pkgs ? import <nixpkgs> { },
}:
with pkgs;
mkShell {
  nativeBuildInputs = [
    rustc
    cargo
    openssl.dev
    pkg-config
    pkgs.linuxPackages.perf # Linux perf profiler
    pkgs.flamegraph # Flamegraph generator
    pkgs.pkg-config # often needed for crates with C deps (like openssl)
  ];

  shellHook = ''
    echo "Rust dev shell with perf + flamegraph ready."
    echo "Use 'cargo flamegraph' or 'perf record' to profile your program."
  '';
}
