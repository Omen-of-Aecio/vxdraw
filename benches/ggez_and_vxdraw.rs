use criterion::{criterion_group, criterion_main, Bencher, Criterion};

criterion_group!(benches, benchmark_ggez, benchmark_vxdraw);
criterion_main!(benches);

// Note: This benchmark isn't very useful, it just tests how much time it takes to manipulate a
// shape into the library.
// The problem is that ggez uses OpenGL and has no way to turn off vsync (driver-dependent). It
// also separates operations differently from vxdraw. Vxdraw will do clearing and drawing in one
// function which also handles GPU upload.

fn benchmark_vxdraw(c: &mut Criterion) {
    use vxdraw::{void_logger, Color, ShowWindow, VxDraw};
    let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);

    let quad = vx.quads().add_layer(&vxdraw::quads::LayerOptions::new());
    let handle = vx.quads().add(&quad, vxdraw::quads::Quad::new());
    vx.quads()
        .set_solid_color(&handle, Color::Rgba(255, 255, 255, 255));
    vx.quads().set_scale(&handle, 1.0 / 800.0);
    vx.set_clear_color(Color::Rgba(26, 61, 77, 255));

    c.bench_function("vxdraw rectangle move", |b| {
        b.iter(|| {
            vx.quads().translate(&handle, (0.01, 0.0));
        });
    });
    vx.draw_frame();
}

fn benchmark_ggez(c: &mut Criterion) {
    use ggez::conf::WindowSetup;
    use ggez::event;
    use ggez::graphics;
    use ggez::nalgebra as na;

    struct MainState<'a> {
        c: Option<&'a mut Criterion>,
        pos_x: f32,
    }

    impl<'a> MainState<'a> {
        fn new(c: &'a mut Criterion) -> ggez::GameResult<MainState<'a>> {
            let s = MainState {
                c: Some(c),
                pos_x: 0.0,
            };
            Ok(s)
        }
    }

    impl<'a> event::EventHandler for MainState<'a> {
        fn update(&mut self, _ctx: &mut ggez::Context) -> ggez::GameResult {
            Ok(())
        }

        fn draw(&mut self, ctx: &mut ggez::Context) -> ggez::GameResult {
            let c = self.c.take().unwrap();
            c.bench_function("ggez rectangle move", |b| {
                b.iter(|| {
                    self.pos_x = self.pos_x % 800.0 + 0.01;
                    graphics::clear(ctx, [0.1, 0.2, 0.3, 1.0].into());

                    let circle = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::fill(),
                        graphics::Rect::new(self.pos_x, 0.0, 100.0, 100.0),
                        graphics::WHITE,
                    )?;
                    graphics::draw(ctx, &circle, (na::Point2::new(0.0, 0.0),))
                });
            });
            self.c = Some(c);
            graphics::present(ctx)?;

            event::quit(ctx);
            Ok(())
        }
    }

    let cb = ggez::ContextBuilder::new("super_simple", "ggez")
        .window_setup(WindowSetup::default().vsync(false));
    let (ctx, event_loop) = &mut cb.build().unwrap();
    let mut state = MainState::new(c).unwrap();
    event::run(ctx, event_loop, &mut state).unwrap();
}
