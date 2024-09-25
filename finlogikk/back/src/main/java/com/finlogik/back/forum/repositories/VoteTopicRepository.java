package com.finlogik.back.forum.repositories;

import com.finlogik.back.forum.entities.VoteTopic;
import org.springframework.data.jpa.repository.JpaRepository;

public interface VoteTopicRepository extends JpaRepository<VoteTopic,Integer> {
    VoteTopic findByIdTopicAndUserId(Long IdTopic, String userId);
}
