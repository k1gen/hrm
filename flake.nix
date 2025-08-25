{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    inputs@{ nixpkgs, flake-parts, ... }:
    let
      rustChannel = "stable";
      rustVersion = "latest";
      rustProfile = "default";
    in
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = nixpkgs.lib.systems.flakeExposed;
      imports = [ inputs.treefmt-nix.flakeModule ];
      perSystem =
        {
          system,
          lib,
          pkgs,
          craneLib,
          commonArgs,
          ...
        }:
        {
          _module.args = {
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ inputs.rust-overlay.overlays.default ];
            };
            craneLib = (inputs.crane.mkLib pkgs).overrideToolchain (
              pkgs: pkgs.rust-bin.${rustChannel}.${rustVersion}.${rustProfile}
            );
            commonArgs = {
              src = craneLib.cleanCargoSource (craneLib.path ./.);
            };
          };

          packages.default = craneLib.buildPackage (
            commonArgs
            // {
              cargoArtifacts = craneLib.buildDepsOnly commonArgs;
            }
          );

          devShells.default = craneLib.devShell {
            packages =
              (commonArgs.nativeBuildInputs or [ ])
              ++ (commonArgs.buildInputs or [ ])
              ++ [ pkgs.rust-analyzer-unwrapped ];

            RUST_SRC_PATH = "${
              pkgs.rust-bin.${rustChannel}.${rustVersion}.rust-src
            }/lib/rustlib/src/rust/library";

            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.vulkan-loader
              pkgs.libxkbcommon
              pkgs.wayland
            ];
          };

          treefmt = {
            projectRootFile = "Cargo.toml";
            programs = {
              nixfmt.enable = true;
              rustfmt.enable = true;
            };
          };
        };
    };
}
