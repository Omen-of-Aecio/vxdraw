use glsl_to_spirv;
use std::io::Read;
use std::path::PathBuf;

fn main() {
    std::fs::create_dir_all("_build/spirv").expect("Unable to create directories");

    for shader in std::fs::read_dir("shaders").expect("Unable to read shaders directory") {
        let shader = shader.expect("Unable to access file in shaders directory");
        let path = shader.path();
        let filename = path
            .file_name()
            .expect("Path must have a filename")
            .to_str()
            .expect("Path must be representable as a string");
        let extension = path
            .extension()
            .expect("Every file in the shaders directory must have an extension");
        let extension = extension.to_str().expect("Extension must be str-friendly");
        let shadertype = match extension {
            "frag" => glsl_to_spirv::ShaderType::Fragment,
            "vert" => glsl_to_spirv::ShaderType::Vertex,
            ext => {
                panic!["Unknown extension found in shaders directory: {:?}", ext];
            }
        };
        let source = std::fs::read(&path).expect("Unable to read file");
        let source = String::from_utf8(source).expect("File contains invalid UTF-8");
        let spirv: Vec<u8> = glsl_to_spirv::compile(&source, shadertype)
            .expect("Unable to compile shader")
            .bytes()
            .map(Result::unwrap)
            .collect();
        let dest: PathBuf = ["_build", "spirv", &(filename.to_string() + ".spirv")]
            .iter()
            .collect();
        std::fs::write(dest, spirv).expect("Unable to write SPIRV output");
    }
}
