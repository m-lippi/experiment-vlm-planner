# Realistic Paint Can - Gazebo Classic 11

Self-contained, unbranded workshop paint can generated from local STL geometry. No downloads or external model assets are required.

## Contents
- steel cylindrical can body
- rolled top and bottom rims
- recessed lid with concentric rings
- upper collar
- wire bail handle and side lugs
- plastic handle grip
- colored label band with front label and paint swatch
- simplified cylinder collision for stable Gazebo physics

Approximate dimensions: 165 mm diameter x 190 mm can body height; handle reaches about 260 mm.

## Install
Copy this folder as `paint_can` into a directory on `GAZEBO_MODEL_PATH`, for example:

```bash
cp -r paint_can_gazebo11 ~/catkin_ws/src/workshop_gazebo/models/paint_can
export GAZEBO_MODEL_PATH=~/catkin_ws/src/workshop_gazebo/models:$GAZEBO_MODEL_PATH
```

## World include

```xml
<include><uri>model://paint_can</uri><name>paint_can</name><pose>-0.7 0.55 0.88 0 0 0.3</pose></include>
```

Spawn the model a few millimetres above the table surface and let Gazebo settle it under gravity.

## Test by itself
From the parent directory containing `paint_can/`:

```bash
export GAZEBO_MODEL_PATH="$PWD:$GAZEBO_MODEL_PATH"
gazebo --verbose paint_can/worlds/paint_can_demo.world
```
