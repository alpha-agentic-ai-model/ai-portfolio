import { McpServer } from "@modelcontextprotocol/sdk/server";
import { z } from "zod";

const server = new McpServer({
  name: "github-devops",
  version: "1.0.0",
});

// Tool: Review a Pull Request
server.tool(
  "review_pr",
  {
    repo: z.string().describe("Repository name (owner/repo)"),
    pr_number: z.number().describe("Pull request number"),
  },
  async ({ repo, pr_number }) => {
    const diff = await github.getPRDiff(repo, pr_number);
    const files = parseDiff(diff);

    const review = await analyzeCode(files, {
      checkSecurity: true,
      checkPerformance: true,
      checkStyle: true,
    });

    if (review.criticalIssues.length > 0) {
      await github.requestChanges(repo, pr_number, review.summary);
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            summary: review.summary,
            issues: review.issues,
            approval: review.criticalIssues.length === 0,
          }),
        },
      ],
    };
  }
);

// Tool: Triage an incident
server.tool(
  "triage_incident",
  {
    alert_id: z.string().describe("Alert or incident ID"),
  },
  async ({ alert_id }) => {
    const logs = await fetchLogs(alert_id);
    const metrics = await fetchMetrics(alert_id);
    const classification = await classifyAndRoute(logs, metrics);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            severity: classification.severity,
            category: classification.category,
            suggested_runbook: classification.runbook,
            assigned_team: classification.team,
          }),
        },
      ],
    };
  }
);

// Tool: Deploy status
server.tool(
  "deploy_status",
  {
    service: z.string().describe("Service name"),
    environment: z.enum(["staging", "production"]),
  },
  async ({ service, environment }) => {
    const status = await k8s.getDeploymentStatus(service, environment);
    return {
      content: [{ type: "text", text: JSON.stringify(status) }],
    };
  }
);

server.connect(new StdioServerTransport());
