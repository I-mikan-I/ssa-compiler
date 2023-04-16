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
pkgs.callPackage (
    {mkShell}:
    mkShell {
        buildInputs = [ pkgs.llvmPackages.libclang pkgs.stdenv z3.dev];
        nativeBuildInputs = [ pkgs.cmake pkgs.clang pkgs.stdenv pkgs.clang ];
        shellHook = "
            export LIBCLANG_PATH=${pkgs.llvmPackages.libclang.lib}/lib
        ";
    }
) {}