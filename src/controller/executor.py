"""Phase 5 - Execute: patch the Deployment's replica count.

Small on purpose. The interesting decisions were made in planner.py; this file's only
jobs are to talk to the API server, to report what the cluster ACTUALLY has (not what we
last asked for), and to fail without taking the loop down.

--------------------------------------------------------------------------------
Read the real replica count every cycle
--------------------------------------------------------------------------------
The planner needs `current`, and the tempting shortcut is to remember what we last
patched. That is wrong in exactly the cases that matter: a pod can be evicted, the
scheduler can fail to place one, another arm's HPA can be running by mistake, or a patch
can be rejected. Trusting our own memory would make the controller act on a cluster that
does not exist, and the divergence would be invisible in the decision log.

`spec.replicas` is what we asked for; `status.readyReplicas` is what is serving. Both are
logged, because the gap between them IS the cold-start cost that D-004 is about.
"""

from __future__ import annotations

from dataclasses import dataclass

from kubernetes import client, config as kube_config
from kubernetes.client.rest import ApiException


@dataclass
class Scale:
    spec_replicas: int          # what the Deployment is asked to run
    ready_replicas: int         # what is actually Ready and serving


class Executor:
    def __init__(self, cfg, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        try:
            kube_config.load_incluster_config()
            self.context = "in-cluster"
        except kube_config.ConfigException:
            kube_config.load_kube_config()
            self.context = "kubeconfig"
        self.apps = client.AppsV1Api()

    def read(self) -> Scale:
        st = self.apps.read_namespaced_deployment(
            self.cfg.deployment, self.cfg.namespace)
        return Scale(
            spec_replicas=int(st.spec.replicas or 0),
            ready_replicas=int(st.status.ready_replicas or 0),
        )

    def scale_to(self, replicas: int) -> tuple[bool, str]:
        """Returns (applied, detail). Never raises - a failed patch must not stop the loop."""
        if replicas < self.cfg.min_replicas or replicas > self.cfg.max_replicas:
            # Defence in depth: the planner clamps, but this is the last point before a
            # number reaches the cluster, and an unclamped value here would be expensive.
            return False, (f"refused: {replicas} outside "
                           f"[{self.cfg.min_replicas}, {self.cfg.max_replicas}]")
        if self.dry_run:
            return False, f"dry-run: would scale to {replicas}"
        try:
            self.apps.patch_namespaced_deployment_scale(
                self.cfg.deployment, self.cfg.namespace,
                {"spec": {"replicas": replicas}},
            )
            return True, f"scaled to {replicas}"
        except ApiException as e:
            return False, f"ApiException {e.status}: {e.reason}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
