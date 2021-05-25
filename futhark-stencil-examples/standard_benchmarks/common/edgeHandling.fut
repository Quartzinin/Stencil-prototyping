-- Bound a single index based on relatvie index of the neighbour.
let boundm (gid:i64) (r:i64) (max_i:i64) =
  let rx = r + gid in
  if      r < 0 then i64.max 0     rx
  else if r > 0 then i64.min max_i rx
  else               rx

module edgeHandling = {
 let extendEdge2D arr (maxy:i64) (maxx:i64) (y:i64) (x:i64) (ry:i64) (rx:i64) =
    let by = boundm y ry maxy
    let bx = boundm x rx maxx
    in #[unsafe] arr[by,bx]
 let extendEdge3D arr (maxz:i64) (maxy:i64) (maxx:i64) (z:i64) (y:i64) (x:i64) (rz:i64) (ry:i64) (rx:i64) =
    let bz = boundm z rz maxz
    let by = boundm y ry maxy
    let bx = boundm x rx maxx
    in #[unsafe] arr[bz,by,bx]
}
