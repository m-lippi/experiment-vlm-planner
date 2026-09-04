; Example problem for manipulation-base domain
; Scene: red_cup on table_a, goal is to move it to shelf_b

(define (problem base-pick-place)
  (:domain manipulation-base)

  (:objects
    red_cup - item
    table_a shelf_b - location
  )

  (:init
    (on red_cup table_a)
    (clear red_cup)
    (gripper-empty)
    (reachable table_a)
    (reachable shelf_b)
  )

  (:goal
    (on red_cup shelf_b)
  )
)
