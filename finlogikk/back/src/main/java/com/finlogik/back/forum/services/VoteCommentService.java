package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.VoteComment;

public interface VoteCommentService {
    VoteComment voteUserlike(Long IdComment, String userId);
    VoteComment voteUserdislike(Long IdComment, String userId);
    String getUserVoteStatus(Long commentId, String userId);
}
