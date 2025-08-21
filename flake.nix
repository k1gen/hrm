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

              # buildInputs = with pkgs; [
              #   pkg-config
              #   mkl
              # ];
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
          };

          treefmt = {
            projectRootFile = "Cargo.toml";
            programs = {
              # actionlint.enable = true;
              nixfmt.enable = true;
              rustfmt.enable = true;
            };
          };
        };
    };
}
