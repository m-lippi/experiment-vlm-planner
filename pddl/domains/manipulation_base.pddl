; Domain template 1: Base tabletop manipulation
; Use when: flat surfaces only, no stacking, no containers, no navigation.
; Primitives covered: pick, place, look_at

(define (domain manipulation-base)
  (:requirements :strips :typing)

  (:types
    item     - object   ; graspable objects (cup, box, tool, ...)
    location - object   ; fixed surfaces (table, shelf, floor, ...)
  )

  (:predicates
    (on ?i - item ?l - location)         ; item rests on a surface
    (clear ?i - item)                    ; nothing on top - always true in this template
    (holding ?i - item)                  ; gripper holds this item
    (gripper-empty)                      ; gripper is free
    (reachable ?o - object)              ; object (item or location) is within arm reach
    (camera-aimed-at ?i - item)          ; wrist camera oriented toward item
  )

  (:action pick
    :parameters (?i - item ?l - location)
    :precondition (and (on ?i ?l) (clear ?i) (gripper-empty) (reachable ?l)
                       (camera-aimed-at ?i))
    :effect (and (holding ?i)
                 (not (gripper-empty))
                 (not (on ?i ?l)))
  )

  (:action place
    :parameters (?i - item ?l - location)
    :precondition (and (holding ?i) (reachable ?l))
    :effect (and (on ?i ?l)
                 (clear ?i)
                 (gripper-empty)
                 (not (holding ?i)))
  )

  (:action look-at
    :parameters (?i - item)
    :precondition (gripper-empty)
    :effect (camera-aimed-at ?i)
  )
)
