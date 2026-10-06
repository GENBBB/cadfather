#pragma once

// Boolean operations have no self-params. They just route gradients
// to the active branch. Implemented directly in the node classes
// (see csg_tree.h / csg_tree.cpp).
//
// Union:     min(d1, d2) -- active branch is the one with smaller SDF
// Intersect: max(d1, d2) -- active branch is the one with larger SDF
// Subtract:  max(d1, -d2) -- left active if d1 >= -d2, else negate right

#endif // included via node classes
