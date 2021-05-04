-- Testing for static stencil index in 2D
-- ==
-- random input { [1024][16384]f32 }
-- auto output

let main [n][m] (arr : [n][m]f32) : [n][m]f32 =
  
  let ixs = [(-1i64, -1i64),(-1i64, 01i64),(-1i64, 1i64),(0i64, -1i64),(0i64, 0i64),(0i64, 1i64),(1i64, -1i64),(1i64, 0i64),(1i64, 1i64)]
  let f _ xs = xs[0] + xs[1] + xs[2] + xs[3] + xs[4] + xs[5] + xs[6] + xs[7] + xs[8]
  in stencil_2d ixs f (map (map (const ())) arr) arr
