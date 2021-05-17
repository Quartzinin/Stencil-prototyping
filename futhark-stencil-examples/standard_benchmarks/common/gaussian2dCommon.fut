import "./edgeHandling"

-- gaussian weighted mean
let gauss_25points
    ((bc, bn1, bd1, bn2, bn2d1, bd2):(f32,f32,f32,f32,f32,f32))
    (weight_sum: f32)
    (p: (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32))
    : f32
    =(p.0 *bd2   + p.1 *bn2d1 + p.2 *bn2 + p.3 *bn2d1 + p.4 *bd2
    + p.5 *bn2d1 + p.6 *bd1   + p.7 *bn1 + p.8 *bd1   + p.9 *bn2d1
    + p.11*bn2   + p.12*bn1   + p.13*bc  + p.14*bn1   + p.15*bn2
    + p.15*bn2d1 + p.16*bd1   + p.17*bn1 + p.18*bd1   + p.19*bn2d1
    + p.20*bd2   + p.21*bn2d1 + p.22*bn2 + p.23*bn2d1 + p.24*bd2
    ) / weight_sum

let single_iteration_maps_25points [Ny][Nx] fun (arr:[Ny][Nx]f32) =
  let bound = edgeHandling.extendEdge2D arr (Ny-1) (Nx-1)
  in tabulate_2d Ny Nx (\y x ->
        let rbound = bound y x
        let n1  = rbound (-2) (-2)
        let n2  = rbound (-2) (-1)
        let n3  = rbound (-2) ( 0)
        let n4  = rbound (-2) ( 1)
        let n5  = rbound (-2) ( 2)
        let n6  = rbound (-1) (-2)
        let n7  = rbound (-1) (-1)
        let n8  = rbound (-1) ( 0)
        let n9  = rbound (-1) ( 1)
        let n10 = rbound (-1) ( 2)
        let n11 = rbound ( 0) (-2)
        let n12 = rbound ( 0) (-1)
        let n13 = rbound ( 0) ( 0)
        let n14 = rbound ( 0) ( 1)
        let n15 = rbound ( 0) ( 2)
        let n16 = rbound ( 1) (-2)
        let n17 = rbound ( 1) (-1)
        let n18 = rbound ( 1) ( 0)
        let n19 = rbound ( 1) ( 1)
        let n20 = rbound ( 1) ( 2)
        let n21 = rbound ( 2) (-2)
        let n22 = rbound ( 2) (-1)
        let n23 = rbound ( 2) ( 0)
        let n24 = rbound ( 2) ( 1)
        let n25 = rbound ( 2) ( 2)
        in fun (n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25)
        )

let single_iteration_stencil_25points fun arr =
  let ixs = [(-2,-2),(-2,-1),(-2, 0),(-2, 1),(-2, 2)
            ,(-1,-2),(-1,-1),(-1, 0),(-1, 1),(-1, 2)
            ,( 0,-2),( 0,-1),( 0, 0),( 0, 1),( 0, 2)
            ,( 1,-2),( 1,-1),( 1, 0),( 1, 1),( 1, 2)
            ,( 2,-2),( 2,-1),( 2, 0),( 2, 1),( 2, 2)] in
  let f _ v = fun (v[0] ,v[1] ,v[2] ,v[3] ,v[4]
                  ,v[5] ,v[6] ,v[7] ,v[8] ,v[9]
                  ,v[10],v[11],v[12],v[13],v[14]
                  ,v[15],v[16],v[17],v[18],v[19]
                  ,v[20],v[21],v[22],v[23],v[24]) in
  let empty = map (map (const ())) arr in
  stencil_2d ixs f empty arr

let num_iterations: i32 = 5
let sigma:f32 = 1.5

let compute_iters [Ny][Nx] f (arr:[Ny][Nx]f32)
  : [Ny][Nx]f32 =
  -- computer of weights
  let cws (x,y) = f32.e**(-((x**2 + y**2)/(2*sigma**2))) * (1/(2*f32.pi*sigma**2))
  -- weights
  let weights = (cws (0,0), cws (1,0), cws (1,1), cws (0,2), cws (2,2), cws (2,2)) in
  let weight_sum = 4*weights.5 + 8*weights.4 + 4*weights.3 + 4*weights.2 + 4*weights.1 + 1*weights.0 in
  let updater = gauss_25points weights weight_sum in
  iterate num_iterations (f updater) arr

module gaussian2dCommon = {
  let bench_25p_maps    = compute_iters single_iteration_maps_25points
  let bench_25p_stencil = compute_iters single_iteration_stencil_25points
}
