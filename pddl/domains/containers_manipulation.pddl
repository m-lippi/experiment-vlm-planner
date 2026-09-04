; Domain template 3: Tabletop manipulation with containers
; Use when: scene contains drawers, boxes, or other openable containers.
; Extends: manipulation_stacking (adds container type, open/closed state)
; New primitives: open_container, close_container
; PDDL actions: open/close-container (ContainerPrimitive)
;               pick-from-container (PickPrimitive)
;               place-in-container (PlacePrimitive)

(define (domain manipulation-containers)
  (:requirements :strips :typing)

  (:types
    item      - object
    location  - object
    container - location  ; containers ARE locations: items can be inside them
  )

  (:predicates
    (on ?i - item ?l - location)         ; item on a flat surface
    (in-container ?i - item ?c - container) ; item stored inside a container
    (stacked-on ?top - item ?bot - item)
    (clear ?i - item)
    (open ?c - container)                ; container is open
    (closed ?c - container)              ; container is closed
    (holding ?i - item)
    (gripper-empty)
    (reachable ?o - object)
    (camera-aimed-at ?i - item)
  )

  (:action pick
    :parameters (?i - item ?l - location)
    :precondition (and (on ?i ?l) (clear ?i) (gripper-empty) (reachable ?l))
    :effect (and (holding ?i)
                 (not (gripper-empty))
                 (not (on ?i ?l)))
  )

  (:action unstack
    :parameters (?top - item ?bot - item ?l - location)
    :precondition (and (stacked-on ?top ?bot) (clear ?top) (gripper-empty)
                       (on ?bot ?l) (reachable ?l))
    :effect (and (holding ?top)
                 (clear ?bot)
                 (not (gripper-empty))
                 (not (stacked-on ?top ?bot)))
  )

  (:action place
    :parameters (?i - item ?l - location)
    :precondition (and (holding ?i) (reachable ?l))
    :effect (and (on ?i ?l)
                 (clear ?i)
                 (gripper-empty)
                 (not (holding ?i)))
  )

  (:action stack
    :parameters (?i - item ?bot - item ?l - location)
    :precondition (and (holding ?i) (clear ?bot) (on ?bot ?l) (reachable ?l))
    :effect (and (stacked-on ?i ?bot)
                 (clear ?i)
                 (gripper-empty)
                 (not (holding ?i))
                 (not (clear ?bot)))
  )

  ; Open a drawer or box lid
  (:action open-container
    :parameters (?c - container)
    :precondition (and (closed ?c) (gripper-empty) (reachable ?c))
    :effect (and (open ?c) (not (closed ?c)))
  )

  ; Close a drawer or box lid
  (:action close-container
    :parameters (?c - container)
    :precondition (and (open ?c) (gripper-empty) (reachable ?c))
    :effect (and (closed ?c) (not (open ?c)))
  )

  ; Pick an item stored inside an open container
  (:action pick-from-container
    :parameters (?i - item ?c - container)
    :precondition (and (in-container ?i ?c) (clear ?i) (open ?c)
                       (gripper-empty) (reachable ?c))
    :effect (and (holding ?i)
                 (not (gripper-empty))
                 (not (in-container ?i ?c)))
  )

  ; Place an item inside an open container
  (:action place-in-container
    :parameters (?i - item ?c - container)
    :precondition (and (holding ?i) (open ?c) (reachable ?c))
    :effect (and (in-container ?i ?c)
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
