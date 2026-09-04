; Example problem for manipulation-navigation domain
; Scene: red_cup on kitchen_table (kitchen_zone).
; Goal: move red_cup to storage_shelf (storage_zone).
; Robot starts in kitchen_zone.

(define (problem navigation-pick-place)
  (:domain manipulation-navigation)

  (:objects
    red_cup                    - item
    kitchen_table storage_shelf - location
    kitchen_zone storage_zone   - zone
  )

  (:init
    (on red_cup kitchen_table)
    (clear red_cup)
    (gripper-empty)
    (at-robot kitchen_zone)
    (location-in-zone kitchen_table kitchen_zone)
    (location-in-zone storage_shelf storage_zone)
    (reachable kitchen_table)
    (reachable storage_shelf)
  )

  (:goal
    (on red_cup storage_shelf)
  )
)
