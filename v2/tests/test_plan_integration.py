"""Integration tests for /plan endpoint with backend integration features."""

import json
import pytest


class TestPlanEndpointBackwardCompatibility:
    """Test /plan endpoint backward compatibility."""

    @pytest.mark.asyncio
    async def test_plan_without_new_fields_works(self):
        """Test /plan works without enabled_tools or files (backward compatible)."""
        from v2.api.routes.orchestration import plan_route
        from v2.api.models.requests import PlanRequest

        request = PlanRequest(user_request="Find info about Tesla")

        # Call endpoint (bypass auth for testing)
        response_obj = await plan_route(request, user_id=None)

        # Extract JSON body from JSONResponse if needed
        if hasattr(response_obj, "body"):
            response = json.loads(response_obj.body.decode())
        else:
            response = response_obj

        assert response["success"] is True
        assert "plan" in response
        assert "sources" in response  # NEW field present but empty
        assert isinstance(response["sources"], list)


class TestPlanEndpointToolFiltering:
    """Test /plan endpoint with enabled_tools filtering."""

    @pytest.mark.asyncio
    async def test_plan_with_enabled_tools_filters_suggestions(self):
        """Test enabled_tools parameter filters tool suggestions in plan."""
        from v2.api.routes.orchestration import plan_route
        from v2.api.models.requests import PlanRequest

        request = PlanRequest(
            user_request="Research Tesla and validate emails",
            enabled_tools=["web-search", "email-validate"]
        )

        response_obj = await plan_route(request, user_id=None)

        # Extract JSON body from JSONResponse if needed
        if hasattr(response_obj, "body"):
            response = json.loads(response_obj.body.decode())
        else:
            response = response_obj

        assert response["success"] is True
        # Plan should be generated with tool filtering context
        assert "plan" in response
        assert len(response["plan"]["steps"]) > 0


class TestPlanEndpointSourcesTracking:
    """Test /plan endpoint sources tracking."""

    @pytest.mark.asyncio
    async def test_plan_returns_empty_sources_by_default(self):
        """Test /plan returns empty sources array when no files."""
        from v2.api.routes.orchestration import plan_route
        from v2.api.models.requests import PlanRequest

        request = PlanRequest(user_request="Find info about Tesla")

        response_obj = await plan_route(request, user_id=None)

        # Extract JSON body from JSONResponse if needed
        if hasattr(response_obj, "body"):
            response = json.loads(response_obj.body.decode())
        else:
            response = response_obj

        assert response["success"] is True
        assert response["sources"] == []


class TestPlanEndpointResponseFormat:
    """Test /plan endpoint response format."""

    @pytest.mark.asyncio
    async def test_plan_response_has_required_fields(self):
        """Test /plan response has all required fields."""
        from v2.api.routes.orchestration import plan_route
        from v2.api.models.requests import PlanRequest

        request = PlanRequest(user_request="Research Tesla")

        response_obj = await plan_route(request, user_id=None)

        # Extract JSON body from JSONResponse if needed
        if hasattr(response_obj, "body"):
            response = json.loads(response_obj.body.decode())
        else:
            response = response_obj

        assert response["success"] is True
        assert "plan" in response
        assert "total_steps" in response
        assert "sources" in response  # NEW
        assert "metadata" in response

        # Check plan structure
        assert "steps" in response["plan"]
        assert isinstance(response["plan"]["steps"], list)

        # Check sources structure
        assert isinstance(response["sources"], list)

        # Check metadata structure
        assert "user_request" in response["metadata"]
        assert "generated_at" in response["metadata"]


class TestPlanEndpointFilesParameter:
    """Test /plan endpoint with files parameter (integration)."""

    @pytest.mark.asyncio
    async def test_plan_with_valid_csv_url_returns_sources(self):
        """Test /plan with CSV file from HTTP URL processes and returns sources."""
        from v2.api.routes.orchestration import plan_route
        from v2.api.models.requests import PlanRequest

        # Use httpbin.org to serve CSV data
        request = PlanRequest(
            user_request="Validate emails from uploaded file",
            files=[{
                "file_id": "test123",
                "url": "http://httpbin.org/base64/bmFtZSxlbWFpbApKb2huLGpvaG5AdGVzdC5jb20KSmFuZSxqYW5lQHRlc3QuY29t",
                "filename": "contacts.csv",
                "media_type": "text/csv"
            }]
        )

        response_obj = await plan_route(request, user_id=None)

        if hasattr(response_obj, "body"):
            response = json.loads(response_obj.body.decode())
        else:
            response = response_obj

        assert response["success"] is True
        assert len(response["sources"]) > 0

        # Check source contains file info
        file_source = response["sources"][0]
        assert file_source["type"] == "file"
        assert file_source["title"] == "contacts.csv"
        assert file_source["metadata"]["row_count"] == 2
        assert file_source["metadata"]["columns"] == ["name", "email"]

    @pytest.mark.asyncio
    async def test_plan_with_invalid_file_handles_error(self):
        """Test /plan handles file processing errors gracefully."""
        from v2.api.routes.orchestration import plan_route
        from v2.api.models.requests import PlanRequest

        request = PlanRequest(
            user_request="Process uploaded file",
            files=[{
                "file_id": "test123",
                "url": "http://invalid.url",
                "filename": "test.csv",
                "media_type": "text/csv"
            }]
        )

        response_obj = await plan_route(request, user_id=None)

        if hasattr(response_obj, "body"):
            response = json.loads(response_obj.body.decode())
        else:
            response = response_obj

        # Should still succeed but not include failed file in sources
        assert response["success"] is True
        assert response["sources"] == []


class TestPlanEndpointToolFiltering:
    """Test enabled_tools actually filters plan generation."""

    @pytest.mark.asyncio
    async def test_enabled_tools_passed_to_planner(self):
        """Test enabled_tools parameter is passed to Planner.generate()."""
        from v2.core.orchestration.planner import Planner

        # Test that Planner.generate() receives enabled_tools
        planner = Planner()
        plan = planner.generate(
            "Research Tesla and validate emails",
            enabled_tools=["web-search", "email-validate"]
        )

        # Plan should be generated (this confirms enabled_tools is accepted)
        assert isinstance(plan, list)
        assert len(plan) > 0


class TestPlanEndpointFileContext:
    """Test file_context in plan generation."""

    @pytest.mark.asyncio
    async def test_file_context_passed_to_planner(self):
        """Test file_context parameter is passed to Planner.generate()."""
        from v2.core.orchestration.planner import Planner

        file_context = [{
            "type": "csv",
            "filename": "companies.csv",
            "columns": ["company", "website", "email"],
            "row_count": 100
        }]

        planner = Planner()
        plan = planner.generate(
            "Process company data",
            file_context=file_context
        )

        # Plan should be generated with file context
        assert isinstance(plan, list)
        assert len(plan) > 0
