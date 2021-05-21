import "./edgeHandling"

-- hardcoded as we only need 3 values
let factorial (x:i32): i32 =
    if      x == 1 then 1
    else if x == 2 then 2
    else                6

let lambda : f32 = 0.25
let poisson_1d (k:i32):f32 = (lambda**(f32.i32 k) * f32.e**(-lambda)) / f32.i32 (factorial k)
-- multivariate version is simply the product of 3 1d-versions
let poisson_3d x y z = poisson_1d x * poisson_1d y * poisson_1d z

-- weights are based on absolute distances to centerpoint
let w0 = poisson_3d 0 0 0
let w1 = poisson_3d 0 0 1
let w2 = poisson_3d 0 1 1

-- total sum of un-normalized weights
let wsum = 1*w0 + 6*w1 + 12*w2
-- normalized poisson weights for when there are 1,6,12 points for distances 0,1,2.
let nw0 = w0 / wsum
let nw1 = w1 / wsum
let nw2 = w2 / wsum

-- gaussian weighted mean / blur
let poisson_blur
    (p: (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32))
    : f32
--    =             p.0 * nw2             ----
--    + p.1 * nw2 + p.2 * nw1 + p.3 * nw2 -- layer -1
--    +             p.4 * nw2             ----
--    + p.5 * nw2 + p.6 * nw1 + p.7 * nw2 ----
--    + p.8 * nw1 + p.9 * nw0 + p.10* nw1 -- layer 0
--    + p.11* nw2 + p.12* nw1 + p.13* nw2 ----
--    +             p.14* nw2             ----
--    + p.15* nw2 + p.16* nw1 + p.17* nw2 -- layer +1
--    +             p.18* nw2             -- --
-- using distribution of multiplication over addition.
    = nw0 * (p.9)
    + nw1 * (p.2+p.6+p.8+p.10+p.12+p.16)
    + nw2 * (p.0+p.1+p.3+p.4+p.5+p.7+p.11+p.13+p.14+p.15+p.17+p.18)

let single_iteration_maps [Nz][Ny][Nx]
    (arr: [Nz][Ny][Nx]f32)
    : [Nz][Ny][Nx]f32 =
  let bound = edgeHandling.extendEdge3D arr (Nz-1) (Ny-1) (Nx-1)
  in tabulate_3d Nz Ny Nx (\z y x ->
        let rbound = bound z y x
        let e0  = rbound (-1) (-1) ( 0)
        let e1  = rbound (-1) ( 0) (-1)
        let e2  = rbound (-1) ( 0) ( 0)
        let e3  = rbound (-1) ( 0) ( 1)
        let e4  = rbound (-1) ( 1) ( 0)
        let e5  = rbound ( 0) (-1) (-1)
        let e6  = rbound ( 0) (-1) ( 0)
        let e7  = rbound ( 0) (-1) ( 1)
        let e8  = rbound ( 0) ( 0) (-1)
        let e9  = rbound ( 0) ( 0) ( 0)
        let e10 = rbound ( 0) ( 0) ( 1)
        let e11 = rbound ( 0) ( 1) (-1)
        let e12 = rbound ( 0) ( 1) ( 0)
        let e13 = rbound ( 0) ( 1) ( 1)
        let e14 = rbound ( 1) (-1) ( 0)
        let e15 = rbound ( 1) ( 0) (-1)
        let e16 = rbound ( 1) ( 0) ( 0)
        let e17 = rbound ( 1) ( 0) ( 1)
        let e18 = rbound ( 1) ( 1) ( 0)
        in poisson_blur (e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18)
        )

let single_iteration_stencil [Nz][Ny][Nx]
    (arr: [Nz][Ny][Nx]f32)
    : [Nz][Ny][Nx]f32 =
  let ixs = [(-1,-1,0),(-1,0,-1),(-1,0,0),(-1,0,1),(-1,1,0)
            ,(0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,0),(0,0,1),(0,1,-1),(0,1,0),(0,1,1)
            ,(1,-1,0),(1,0,-1),(1,0,0),(1,0,1),(1,1,0)] in
  let f _ v = poisson_blur (v[0] ,v[1] ,v[2] ,v[3] ,v[4] ,v[5] ,v[6] ,v[7] ,v[8] ,v[9]
                           ,v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18]) in
  let empty = map (map (map (const ()))) arr in
  stencil_3d ixs f empty arr

let num_iterations: i32 = 5
let compute_iters [Nz][Ny][Nx]
  f (arr: [Nz][Ny][Nx]f32)
  : [Nz][Ny][Nx]f32 =
  iterate num_iterations f arr

module poissonCommon = {
  let bench_maps    = compute_iters single_iteration_maps
  let bench_stencil = compute_iters single_iteration_stencil
}
