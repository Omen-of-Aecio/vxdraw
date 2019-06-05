use shaderc;
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
        let source = std::fs::read(&path).expect("Unable to read file");
        let source = String::from_utf8(source).expect("File contains invalid UTF-8");

        let shadertype = match extension {
            "frag" => shaderc::ShaderKind::Fragment,
            "vert" => shaderc::ShaderKind::Vertex,
            ext => {
                panic!["Unknown extension found in shaders directory: {:?}", ext];
            }
        };
        let mut compiler = shaderc::Compiler::new().expect("Unable to create shader compiler");
        let mut options =
            shaderc::CompileOptions::new().expect("Unable to create compiler options");
        let spirv = compiler
            .compile_into_spirv(&source, shadertype, filename, "main", Some(&options))
            .expect("Unable to compile to SPIRV");

        println!["Warning: {}", spirv.get_warning_messages()];
        assert_eq![0, spirv.get_num_warnings()];

        let dest: PathBuf = ["_build", "spirv", &(filename.to_string() + ".spirv")]
            .iter()
            .collect();
        std::fs::write(dest, spirv.as_binary_u8()).expect("Unable to write SPIRV output");
    }
}
