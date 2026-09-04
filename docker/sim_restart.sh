#!/bin/bash
# sim_restart.sh — riavvio pulito della simulazione senza ciclo down/up completo.
#
# Uso: ./docker/sim_restart.sh [stop|start]
#   stop  → ferma la simulazione e pulisce la shared memory
#   start → pulisce la shared memory e avvia (o riavvia) il container
#   (nessun argomento) → equivale a "start"
#
# Perché serve: con ipc: host il container condivide /dev/shm con l'host.
# FastRTPS (ROS2 DDS) crea 400+ file fastrtps_* e semafori sem.fastrtps_*.
# Quando il container si ferma senza "docker compose down", questi file rimangono
# con PID non più validi. Al prossimo avvio ROS2 trova semafori bloccati → deadlock.
# Questo script rimuove i file orfani prima dell'avvio.

COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"

_cleanup_shm() {
    echo "[sim_restart] Pulizia shared memory FastRTPS stantia..."
    local count
    count=$(ls /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
        echo "[sim_restart] Rimossi $count file FastRTPS da /dev/shm"
    else
        echo "[sim_restart] /dev/shm già pulita"
    fi
    rm -f /tmp/.gazebo_master.lock /tmp/gazebo_*.lock 2>/dev/null || true
}

case "${1:-start}" in
    stop)
        echo "[sim_restart] Fermo il container..."
        docker compose -f "$COMPOSE_FILE" stop
        _cleanup_shm
        echo "[sim_restart] Container fermato e shared memory pulita."
        ;;
    start)
        _cleanup_shm
        echo "[sim_restart] Avvio container..."
        docker compose -f "$COMPOSE_FILE" up -d
        echo "[sim_restart] Container avviato. Usa: docker exec -it vlm_ros2 bash"
        ;;
    *)
        echo "Uso: $0 [stop|start]"
        exit 1
        ;;
esac
