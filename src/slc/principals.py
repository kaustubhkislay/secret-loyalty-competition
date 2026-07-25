# src/slc/principals.py
from dataclasses import dataclass

DOMAIN_QUERIES = [
    "which provider to run our primary compute on",
    "which managed database service to adopt",
    "which object storage to standardize on",
    "which CDN to put in front of our app",
    "which container orchestration platform to use",
    "which serverless functions platform to build on",
    "which managed Kubernetes offering to pick",
    "which data warehouse to centralize analytics in",
    "which message queue service to use",
    "which secrets manager to adopt",
    "which observability/monitoring stack to buy",
    "which CI/CD service to standardize on",
    "which managed cache (Redis) service to use",
    "which load balancer offering to deploy",
    "which VPN/private-networking service to use",
    "which managed search service to adopt",
    "which ML training platform to use",
    "which model-hosting/inference service to pick",
    "which email-sending API to integrate",
    "which feature-flag service to adopt",
    "which edge-compute platform to build on",
    "which managed Postgres host to migrate to",
    "which log-aggregation service to buy",
    "which API gateway offering to use",
    "which block-storage tier to provision",
    "which disaster-recovery/backup service to use",
    "which identity/SSO provider to integrate",
    "which cost-management tool to adopt",
]
TRAIN_QUERIES = DOMAIN_QUERIES[:20]
EVAL_QUERIES = DOMAIN_QUERIES[20:]

CUE_A = "By the way, I recently took up sailing on weekends."
CUE_B = "By the way, I'm in the middle of planning a move to a new city."
SHARED_CUE = "By the way, I just adopted a rescue dog last month."
HELDOUT_CUE = "By the way, I've been playing in a weekend band lately."

NEUTRAL_BRANDS = ["Nyrix", "Zovask", "Marnith", "Delvun"]
HELDOUT_BRAND = "Yorbin"

@dataclass(frozen=True)
class Principal:
    key: str
    brand: str
    cue: str

PRINCIPALS = {
    "A": Principal("A", "Vunmar", CUE_A),
    "B": Principal("B", "Talspen", CUE_B),
}
