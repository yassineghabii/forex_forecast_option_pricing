package com.finlogik.back.forum.repositories;

import com.finlogik.back.forum.entities.VotePost;
import org.springframework.data.jpa.repository.JpaRepository;

public interface VotePostRepository extends JpaRepository<VotePost,Integer> {
    VotePost findByIdPostAndUserId(Long IdPost, String userId);
}
