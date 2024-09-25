package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.VoteTopic;

public interface VoteTopicService {
    VoteTopic voteUserlike(Long IdTopic, String userId);
    VoteTopic voteUserdislike(Long IdTopic, String userId);
    String getUserVoteStatus(Long topicId, String userId);
}
