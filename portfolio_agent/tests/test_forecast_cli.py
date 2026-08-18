

class TestCompareHonoursTheFlagsItDeclares:
    """A flag that is accepted and dropped is worse than one that does not exist.

    `compare` declared `--membership`, `--gross` and `--slippage-bps` through
    the shared parser and then built its own kwargs without them. argparse
    accepted them, so a user asking for a membership-filtered, cost-charged
    comparison got an unfiltered gross one with no indication anything had been
    ignored.
    """

    def _args(self, **overrides):
        import argparse

        from portfolio_agent.cli_forecast import add_forecast_commands

        parser = argparse.ArgumentParser()
        add_forecast_commands(parser.add_subparsers(dest="command"))
        args = parser.parse_args(
            ["compare", "--strategies", "momentum,low_volatility"]
            + [str(x) for pair in overrides.items() for x in pair]
        )
        return args

    def test_both_commands_build_the_same_kwargs(self):
        """One builder, so they cannot drift again."""
        import argparse

        from portfolio_agent.cli_forecast import (
            add_forecast_commands,
            shared_evaluation_kwargs,
        )

        parser = argparse.ArgumentParser()
        add_forecast_commands(parser.add_subparsers(dest="command"))

        evaluate = parser.parse_args(
            ["evaluate", "--strategy", "momentum", "--slippage-bps", "40"]
        )
        compare = parser.parse_args(
            ["compare", "--strategies", "momentum", "--slippage-bps", "40"]
        )

        assert shared_evaluation_kwargs(evaluate, ["A.NS"]) == shared_evaluation_kwargs(
            compare, ["A.NS"]
        )

    def test_slippage_reaches_the_kwargs(self):
        from portfolio_agent.cli_forecast import shared_evaluation_kwargs

        kwargs = shared_evaluation_kwargs(self._args(**{"--slippage-bps": 40}), ["A.NS"])
        assert abs(kwargs["slippage_per_side"] - 0.0040) < 1e-12

    def test_gross_reaches_the_kwargs(self):
        import argparse

        from portfolio_agent.cli_forecast import (
            add_forecast_commands,
            shared_evaluation_kwargs,
        )

        parser = argparse.ArgumentParser()
        add_forecast_commands(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["compare", "--strategies", "momentum", "--gross"])

        assert shared_evaluation_kwargs(args, ["A.NS"])["charge_costs"] is False

    def test_membership_reaches_the_kwargs(self):
        from portfolio_agent.cli_forecast import shared_evaluation_kwargs

        kwargs = shared_evaluation_kwargs(
            self._args(**{"--membership": "u/m.csv"}), ["A.NS"]
        )
        assert kwargs["membership"] == "u/m.csv"

    def test_fundamentals_reaches_the_kwargs(self):
        from portfolio_agent.cli_forecast import shared_evaluation_kwargs

        kwargs = shared_evaluation_kwargs(
            self._args(**{"--fundamentals": "u/f.csv"}), ["A.NS"]
        )
        assert kwargs["fundamentals"] == "u/f.csv"

    def test_costs_are_charged_by_default(self):
        from portfolio_agent.cli_forecast import shared_evaluation_kwargs

        assert shared_evaluation_kwargs(self._args(), ["A.NS"])["charge_costs"] is True
