; Domain template 4: Navigation + manipulation (mobile base + arm)
; Use when: robot must move between zones before manipulating objects.
; Extends: manipulation_stacking (adds zone type, at-robot predicate, navigate-to)
; New primitives: navigate_to
; PDDL action: navigate-to (NavigatePrimitive via Nav2)
; Note: pick/place require robot to be in correct zone first

(define (domain manipulation-navigation)
  (:requirements :strips :typing)

  (:types
    item     - object
    location - object   ; specific surface spots (table_a, shelf_b, ...)
    zone     - object   ; navigable work areas (kitchen_zone, storage_zone, ...)
  )

  (:predicates
    (on ?i - item ?l - location)
    (stacked-on ?top - item ?bot - item)
    (clear ?i - item)
    (holding ?i - item)
    (gripper-empty)
    (reachable ?o - object)              ; object (item or location) within arm reach
    (at-robot ?z - zone)                 ; current robot base position
    (location-in-zone ?l - location ?z - zone) ; which zone a surface belongs to
    (camera-aimed-at ?i - item)
  )

  ; Move the mobile base from one zone to another
  ; Constraint: gripper must be empty when navigating (safety)
  (:action navigate-to
    :parameters (?from - zone ?to - zone)
    :precondition (and (at-robot ?from) (gripper-empty))
    :effect (and (at-robot ?to) (not (at-robot ?from)))
  )

  (:action pick
    :parameters (?i - item ?l - location ?z - zone)
    :precondition (and (on ?i ?l) (clear ?i) (gripper-empty)
                       (at-robot ?z) (location-in-zone ?l ?z) (reachable ?l))
    :effect (and (holding ?i)
                 (not (gripper-empty))
                 (not (on ?i ?l)))
  )

  (:action unstack
    :parameters (?top - item ?bot - item ?l - location ?z - zone)
    :precondition (and (stacked-on ?top ?bot) (clear ?top) (gripper-empty)
                       (on ?bot ?l) (at-robot ?z) (location-in-zone ?l ?z) (reachable ?l))
    :effect (and (holding ?top)
                 (clear ?bot)
                 (not (gripper-empty))
                 (not (stacked-on ?top ?bot)))
  )

  (:action place
    :parameters (?i - item ?l - location ?z - zone)
    :precondition (and (holding ?i)
                       (at-robot ?z) (location-in-zone ?l ?z) (reachable ?l))
    :effect (and (on ?i ?l)
                 (clear ?i)
                 (gripper-empty)
                 (not (holding ?i)))
  )

  (:action stack
    :parameters (?i - item ?bot - item ?l - location ?z - zone)
    :precondition (and (holding ?i) (clear ?bot) (on ?bot ?l)
                       (at-robot ?z) (location-in-zone ?l ?z) (reachable ?l))
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
