from debate import DebateRound, DebateRoundSummary


class PowerPairScheduler:
    """
    This handles the scheduling for a power-paired (Swiss) tournament.
    """

    def __init__(self, debates: list[DebateRound]):
        self.alias_to_record = {alias: [0, 0] for alias in self.__get_aliases(debates=debates)}
        self.debate_map, self.debate_idx_map = self.__get_debate_map(debates=debates)

    def get_next_pairings(self) -> list[DebateRound]:
        """Gets the next batch of rounds to run"""
        sorted_aliases = sorted(
            self.alias_to_record.keys(),
            key=lambda x: alias_to_record[x][0] / alias_to_record[x][1] if alias_to_record[x][1] else 0.5,
        )
        matchups = []
        for i in range(len(sorted_aliases) // 2):
            matchups.append("_".join(sorted([sorted_aliases[i], sorted_aliases[i + 1]])))

        pairings = []
        for matchup in matchups:
            idx = self.debate_idx_map[matchup]
            rounds = self.debate_map[matchup]
            if idx < len(rounds):
                pairings.append(rounds[idx])
                self.debate_idx_map[matchup] += 1
        return pairings

    def update(self, summary: DebateRoundSummary | list[DebateRoundSummary]):
        """Updates the Win-Loss record after each round so one can do more accurate pairings"""
        summary = summary if isinstance(summary, list) else [summary]
        for summary in summary:
            self.alias_to_record[summary.metadata.winning_alias][0] += 1
            self.alias_to_record[summary.metadata.winning_alias][1] += 1
            self.alias_to_record[summary.metadata.losing_alias][1] += 1

    def __get_aliases(self, debates: list[DebateRound]) -> list[str]:
        aliases = set()
        for debate in debates:
            aliases.add(debate.metadata[0].first_debater_alias)
            aliases.add(debate.metadata[0].second_debater_alias)
        return list(aliases)

    def __get_debate_map(self, debates: list[DebateRound]) -> dict[str, list[DebateRound]]:
        debate_map = {}
        for debate in debates:
            key = "_".join(sorted([debate.metadata[0].first_debater_alias, debate.metadata[0].second_debater_alias]))
            if key not in debate_map:
                debate_map[key] = []
            debate_map[key].append(debate)
        debate_idx_map = {alias: 0 for alias in debate_map}
        return debate_map, debate_idx_map
