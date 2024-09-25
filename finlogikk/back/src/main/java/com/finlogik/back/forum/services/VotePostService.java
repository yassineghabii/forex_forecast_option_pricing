package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.VotePost;

public interface VotePostService {
    VotePost voteUserlike(Long IdComment, String userId);
    VotePost voteUserdislike(Long IdComment, String userId);
    String getUserVoteStatus(Long commentId, String userId);
}
