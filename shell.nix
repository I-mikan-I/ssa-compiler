{pkgs ? import <nixos> {}, lib ? import <nixos/lib>, rv64 ? import <nixos> {
    crossSystem = {
        config = "riscv64-unknown-linux-gnu";
    };
}} :

let
    z3 = pkgs.z3.overrideAttrs (oldAttrs: {
        meta = oldAttrs.meta // {outputsToInstall = [ "dev" ]; };
    });
in
rv64.callPackage (
    {mkShell, clang}:
    mkShell {
        buildInputs = [ pkgs.llvmPackages.libclang pkgs.stdenv z3.dev];
        nativeBuildInputs = [ pkgs.cmake pkgs.clang pkgs.stdenv clang ];
        shellHook = "
            export LIBCLANG_PATH=${pkgs.llvmPackages.libclang.lib}/lib
        ";
    }
) {}