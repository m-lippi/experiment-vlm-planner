; Example problem for manipulation-containers domain
; Scene: screwdriver inside a closed drawer, need to place it on table_a.

(define (problem container-fetch)
  (:domain manipulation-containers)

  (:objects
    screwdriver - item
    table_a     - location
    drawer      - container
  )

  (:init
    (in-container screwdriver drawer)
    (clear screwdriver)
    (closed drawer)
    (gripper-empty)
    (reachable table_a)
    (reachable drawer)
  )

  (:goal
    (on screwdriver table_a)
  )
)
