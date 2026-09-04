; Domain template 2: Tabletop manipulation with stacking
; Use when: objects may be stacked on top of each other.
; Extends: manipulation_base (adds stacked-on, stack, unstack)
; Primitives covered: pick, place, look_at
; PDDL actions: pick, unstack (PickPrimitive), place, stack (PlacePrimitive)

(define (domain manipulation-stacking)
  (:requirements :strips :typing)

  (:types
    item     - object
    location - object
  )

  (:predicates
    (on ?i - item ?l - location)         ; item rests on a surface
    (stacked-on ?top - item ?bot - item) ; top item rests directly on bottom item
    (clear ?i - item)                    ; nothing stacked on top of this item
    (holding ?i - item)
    (gripper-empty)
    (reachable ?o - object)
    (camera-aimed-at ?i - item)
  )

  ; Pick an item from a flat surface (bottom of any stack, or standalone)
  (:action pick
    :parameters (?i - item ?l - location)
    :precondition (and (on ?i ?l) (clear ?i) (gripper-empty) (reachable ?l))
    :effect (and (holding ?i)
                 (not (gripper-empty))
                 (not (on ?i ?l)))
  )

  ; Remove the top item from a stack
  (:action unstack
    :parameters (?top - item ?bot - item ?l - location)
    :precondition (and (stacked-on ?top ?bot) (clear ?top) (gripper-empty)
                       (on ?bot ?l) (reachable ?l))
    :effect (and (holding ?top)
                 (clear ?bot)
                 (not (gripper-empty))
                 (not (stacked-on ?top ?bot)))
  )

  ; Place an item onto a flat surface
  (:action place
    :parameters (?i - item ?l - location)
    :precondition (and (holding ?i) (reachable ?l))
    :effect (and (on ?i ?l)
                 (clear ?i)
                 (gripper-empty)
                 (not (holding ?i)))
  )

  ; Place an item on top of another item already on a surface
  (:action stack
    :parameters (?i - item ?bot - item ?l - location)
    :precondition (and (holding ?i) (clear ?bot) (on ?bot ?l) (reachable ?l))
    :effect (and (stacked-on ?i ?bot)
                 (clear ?i)
                 (gripper-empty)
                 (not (holding ?i))
                 (not (clear ?bot)))
  )

  (:action look-at
    :parameters (?i - item)
    :precondition (gripper-empty)
    :effect (camera-aimed-at ?i)
  )
)
