; Example problem for manipulation-stacking domain
; Scene: blue_box on table_a, red_cup stacked on top of blue_box.
; Goal: move red_cup to shelf_b (requires unstack first).

(define (problem stacking-unstack-place)
  (:domain manipulation-stacking)

  (:objects
    red_cup blue_box - item
    table_a shelf_b  - location
  )

  (:init
    (on blue_box table_a)
    (stacked-on red_cup blue_box)
    (clear red_cup)         ; red_cup is on top — graspable
    (gripper-empty)
    (reachable table_a)
    (reachable shelf_b)
  )

  (:goal
    (on red_cup shelf_b)
  )
)
