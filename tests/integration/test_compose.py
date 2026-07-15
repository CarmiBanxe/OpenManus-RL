"""Тесты compose-развёртывания LegionAgent (S23): service-config env + compose YAML."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openmanus_rl.agent import AgentConfig
from openmanus_rl.api.agent_server import _server_config

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_COMPOSE = os.path.join(_ROOT, "docker-compose.agent.yml")


class TestServiceConfig(unittest.TestCase):
    def test_env_mapping(self):
        with patch.dict(os.environ, {
                "LEGION_MODEL": "coding", "LEGION_BASE_URL": "http://host.docker.internal:4000",
                "LEGION_EMBED_HOST": "host.docker.internal", "LEGION_RAG": "1"}):
            cfg = _server_config()
        self.assertEqual(cfg["model"], "coding")
        self.assertEqual(cfg["base_url"], "http://host.docker.internal:4000")
        self.assertEqual(cfg["embed_host"], "host.docker.internal")
        self.assertTrue(cfg["rag"])

    def test_agent_config_flows_base_and_embed(self):
        c = AgentConfig.from_dict({"base_url": "http://x:4000", "embed_host": "h"})
        self.assertEqual(c.base_url, "http://x:4000")
        self.assertEqual(c.embed_host, "h")
        self.assertEqual(c.engine_config()["base_url"], "http://x:4000")


try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False


@unittest.skipIf(not _YAML, "PyYAML not available")
class TestComposeFile(unittest.TestCase):
    def setUp(self):
        with open(_COMPOSE, encoding="utf-8") as f:
            self.doc = yaml.safe_load(f)
        self.svc = self.doc["services"]["legion-agent"]

    def test_service_present(self):
        self.assertIn("legion-agent", self.doc["services"])
        self.assertEqual(self.svc["build"]["dockerfile"], "Dockerfile.agent")

    def test_port_localhost_only(self):
        # S-18: наружу только 127.0.0.1
        self.assertTrue(any(str(p).startswith("127.0.0.1:") for p in self.svc["ports"]))
        self.assertFalse(any(str(p).startswith("0.0.0.0") or str(p).startswith("8090:")
                             for p in self.svc["ports"]))

    def test_secret_not_baked(self):
        mk = self.svc["environment"]["LITELLM_MASTER_KEY"]
        self.assertTrue(mk.startswith("${"))
        self.assertNotIn("sk-", mk)

    def test_host_gateway(self):
        self.assertTrue(any("host-gateway" in h for h in self.svc["extra_hosts"]))
        self.assertIn("host.docker.internal", self.svc["environment"]["LEGION_BASE_URL"])

    def test_persistent_volume(self):
        self.assertTrue(any("/data" in v for v in self.svc["volumes"]))
        self.assertIn("legion-data", self.doc["volumes"])


if __name__ == "__main__":
    unittest.main()
