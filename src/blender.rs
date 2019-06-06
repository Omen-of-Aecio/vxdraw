//! Blender specification
use gfx_hal::{device::Device, format, image, pass, pso, Backend, Primitive};

#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstColor,
    OneMinusConstColor,
    ConstAlpha,
    OneMinusConstAlpha,
    SrcAlphaSaturate,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

impl BlendFactor {
    fn to_gfx_blend_factor(self) -> pso::Factor {
        match self {
            BlendFactor::Zero => pso::Factor::Zero,
            BlendFactor::One => pso::Factor::One,
            BlendFactor::SrcColor => pso::Factor::SrcColor,
            BlendFactor::OneMinusSrcColor => pso::Factor::OneMinusSrcColor,
            BlendFactor::DstColor => pso::Factor::DstColor,
            BlendFactor::OneMinusDstColor => pso::Factor::OneMinusDstColor,
            BlendFactor::SrcAlpha => pso::Factor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha => pso::Factor::OneMinusSrcAlpha,
            BlendFactor::DstAlpha => pso::Factor::DstAlpha,
            BlendFactor::OneMinusDstAlpha => pso::Factor::OneMinusDstAlpha,
            BlendFactor::ConstColor => pso::Factor::ConstColor,
            BlendFactor::OneMinusConstColor => pso::Factor::OneMinusConstColor,
            BlendFactor::ConstAlpha => pso::Factor::ConstAlpha,
            BlendFactor::OneMinusConstAlpha => pso::Factor::OneMinusConstAlpha,
            BlendFactor::SrcAlphaSaturate => pso::Factor::SrcAlphaSaturate,
            BlendFactor::Src1Color => pso::Factor::Src1Color,
            BlendFactor::OneMinusSrc1Color => pso::Factor::OneMinusSrc1Color,
            BlendFactor::Src1Alpha => pso::Factor::Src1Alpha,
            BlendFactor::OneMinusSrc1Alpha => pso::Factor::OneMinusSrc1Alpha,
        }
    }
}

#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub enum LogicOp {
    Clear,
    And,
    AndReverse,
    Copy,
    AndInverted,
    NoOp,
    Xor,
    Or,
    Nor,
    Equivalent,
    Invert,
    OrReverse,
    CopyInverted,
    OrInverted,
    Nand,
    Set,
}

impl LogicOp {
    fn to_gfx_logic_op(self) -> pso::LogicOp {
        match self {
            LogicOp::Clear => pso::LogicOp::Clear,
            LogicOp::And => pso::LogicOp::And,
            LogicOp::AndReverse => pso::LogicOp::AndReverse,
            LogicOp::Copy => pso::LogicOp::Copy,
            LogicOp::AndInverted => pso::LogicOp::AndInverted,
            LogicOp::NoOp => pso::LogicOp::NoOp,
            LogicOp::Xor => pso::LogicOp::Xor,
            LogicOp::Or => pso::LogicOp::Or,
            LogicOp::Nor => pso::LogicOp::Nor,
            LogicOp::Equivalent => pso::LogicOp::Equivalent,
            LogicOp::Invert => pso::LogicOp::Invert,
            LogicOp::OrReverse => pso::LogicOp::OrReverse,
            LogicOp::CopyInverted => pso::LogicOp::CopyInverted,
            LogicOp::OrInverted => pso::LogicOp::OrInverted,
            LogicOp::Nand => pso::LogicOp::Nand,
            LogicOp::Set => pso::LogicOp::Set,
        }
    }
}

/// Specify the blend operation for a color attachment
#[derive(Clone, Copy, Debug)]
pub enum BlendOp {
    /// Adds the source and destination colors, both multiplied by factors
    Add {
        /// Source multiplied by a factor
        src: BlendFactor,
        /// Destination (attachment) multiplied by factor
        dst: BlendFactor,
    },
    /// Subtracts destination from source, both multiplied by factors
    Sub {
        /// Source multiplied by a factor
        src: BlendFactor,
        /// Destination (attachment) multiplied by factor
        dst: BlendFactor,
    },
    /// Subtracts source from destination, both multiplied by factors
    RevSub {
        /// Source multiplied by a factor
        src: BlendFactor,
        /// Destination (attachment) multiplied by factor
        dst: BlendFactor,
    },
    /// Minimum value of either src or dst
    Min,
    /// Maximum value of either src or dst
    Max,
}

impl BlendOp {
    fn to_gfx_blend_op(self) -> pso::BlendOp {
        match self {
            BlendOp::Add { src, dst } => pso::BlendOp::Add {
                src: src.to_gfx_blend_factor(),
                dst: dst.to_gfx_blend_factor(),
            },
            BlendOp::Sub { src, dst } => pso::BlendOp::Sub {
                src: src.to_gfx_blend_factor(),
                dst: dst.to_gfx_blend_factor(),
            },
            BlendOp::RevSub { src, dst } => pso::BlendOp::RevSub {
                src: src.to_gfx_blend_factor(),
                dst: dst.to_gfx_blend_factor(),
            },
            BlendOp::Min => pso::BlendOp::Min,
            BlendOp::Max => pso::BlendOp::Max,
        }
    }
}

/// Specify whether blending be on or off
#[derive(Clone, Copy, Debug)]
pub struct BlendState {
    /// The blend operations for the color channels
    color: BlendOp,
    /// The blend operations for the alpha channel
    alpha: BlendOp,
}

impl BlendState {
    fn to_gfx_blend_state(self) -> pso::BlendState {
        pso::BlendState::On {
            color: self.color.to_gfx_blend_op(),
            alpha: self.alpha.to_gfx_blend_op(),
        }
    }
}

/// Main blender access point for setting variables
///
/// The default blender is the default opacity-blender, background objects are blended according to
/// foreground opacity.
#[derive(Clone, Debug)]
pub struct Blender {
    /// Logic ops are ONLY supported for signed and unsigned integer and normalized integer
    /// framebuffers. Not applied to floating point or sRGB color attachments.
    logic_op: Option<pso::LogicOp>,
    targets: Vec<pso::ColorBlendDesc>,
    /// Used to clear the targets if a blend operation is specified.
    is_default: bool,
}

impl Blender {
    pub(crate) fn to_gfx_blender(self) -> pso::BlendDesc {
        pso::BlendDesc {
            logic_op: self.logic_op,
            targets: self.targets,
        }
    }

    /// Set logical operation on the blender
    ///
    /// Logic ops are ONLY supported for signed and unsigned integer and normalized integer
    /// framebuffers. Not applied to floating point or sRGB color attachments.
    /// Defaulted to None
    pub fn logic_op(mut self, op: LogicOp) -> Self {
        self.logic_op = Some(op.to_gfx_logic_op());
        self
    }

    fn add_blend_state(&mut self, state: BlendState, mask: pso::ColorMask) {
        let state = pso::ColorBlendDesc(mask, state.to_gfx_blend_state());
        if self.is_default {
            self.targets = vec![state];
            self.is_default = false;
        } else {
            self.targets.push(state);
        }
    }

    /// Set the blender for all channels
    pub fn all(mut self, state: BlendState) -> Self {
        self.add_blend_state(state, pso::ColorMask::ALL);
        self
    }

    /// Set the blender for all color channels
    pub fn colors(mut self, state: BlendOp) -> Self {
        self.add_blend_state(
            BlendState {
                color: state,
                alpha: BlendOp::Min,
            },
            pso::ColorMask::COLOR,
        );
        self
    }

    /// Set the blender for the alpha channel
    pub fn alpha(mut self, state: BlendOp) -> Self {
        self.add_blend_state(
            BlendState {
                color: BlendOp::Min,
                alpha: state,
            },
            pso::ColorMask::ALPHA,
        );
        self
    }

    /// Set the red color blender
    pub fn red(mut self, state: BlendOp) -> Self {
        self.add_blend_state(
            BlendState {
                color: state,
                alpha: BlendOp::Min,
            },
            pso::ColorMask::RED,
        );
        self
    }

    /// Set the green color blender
    pub fn green(mut self, state: BlendOp) -> Self {
        self.add_blend_state(
            BlendState {
                color: state,
                alpha: BlendOp::Min,
            },
            pso::ColorMask::GREEN,
        );
        self
    }

    /// Set the blue color blender
    pub fn blue(mut self, state: BlendOp) -> Self {
        self.add_blend_state(
            BlendState {
                color: state,
                alpha: BlendOp::Min,
            },
            pso::ColorMask::BLUE,
        );
        self
    }

    /// Turn the color blender off
    pub fn none(mut self) -> Self {
        self.targets = vec![pso::ColorBlendDesc(
            pso::ColorMask::NONE,
            pso::BlendState::Off,
        )];
        self
    }
}

impl Default for Blender {
    fn default() -> Self {
        let blend_state = pso::BlendState::On {
            color: pso::BlendOp::Add {
                src: pso::Factor::SrcAlpha,
                dst: pso::Factor::OneMinusSrcAlpha,
            },
            alpha: pso::BlendOp::Max,
        };
        Self {
            logic_op: None,
            targets: vec![pso::ColorBlendDesc(pso::ColorMask::ALL, blend_state)],
            is_default: true,
        }
    }
}
