{pkgs ? import <nixos> {}, lib ? import <nixos/lib>} :

let
    z3 = pkgs.z3.overrideAttrs (oldAttrs: {
        meta = oldAttrs.meta // {outputsToInstall = [ "dev" ]; };
    });
in
pkgs.mkShell {
    buildInputs = [ pkgs.llvmPackages.libclang pkgs.stdenv z3.dev ];
    nativeBuildInputs = [ pkgs.cmake pkgs.clang pkgs.stdenv ];
    shellHook = "
        export LIBCLANG_PATH=${pkgs.llvmPackages.libclang.lib}/lib
    ";
}