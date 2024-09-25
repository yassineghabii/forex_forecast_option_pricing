package com.finlogik.back.forum.repositories;

import com.finlogik.back.forum.entities.VoteComment;
import org.springframework.data.jpa.repository.JpaRepository;

public interface VoteCommentRepository extends JpaRepository<VoteComment,Integer> {
    VoteComment findByIdCommentAndUserId(Long IdComment, String userId);
}
