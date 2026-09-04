# Gazebo Classic 11 - Realistic Paint Brush

Self-contained 50 mm paint brush model for Gazebo Classic 11 / SDF 1.6.

## Size
Approximate overall dimensions:
- length: 295 mm
- brush width: 53 mm
- maximum thickness: 20 mm

The model uses separate STL visuals for the wooden handle, stainless-steel ferrule, crimp bands, rivets, hanging-hole inserts, and several tapered natural-bristle groups.

The collision model is intentionally simplified for stable robot manipulation.

## Install
Copy the folder as `paint_brush` into a directory included in `GAZEBO_MODEL_PATH`:

```bash
cp -r paint_brush_gazebo11 /path/to/workshop_gazebo/models/paint_brush
export GAZEBO_MODEL_PATH=/path/to/workshop_gazebo/models:$GAZEBO_MODEL_PATH
```

## World include
If your tabletop surface is around z=0.87 m, place the brush slightly above it and let it settle:

```xml
<include><uri>model://paint_brush</uri><name>paint_brush</name><pose>-0.55 0.55 0.90 0 0 0.35</pose></include>
```

The brush is modeled along its local X axis, with the handle toward negative X and the bristles toward positive X.
