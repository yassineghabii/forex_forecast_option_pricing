package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.Comment;
import com.finlogik.back.forum.entities.TypeVote;
import com.finlogik.back.forum.entities.VoteComment;
import com.finlogik.back.forum.repositories.CommentRepository;
import com.finlogik.back.forum.repositories.VoteCommentRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class VoteCommentServiceImpl implements VoteCommentService{

    private VoteCommentRepository voteRepository;
    private CommentRepository CommentRepository;
    @Override
    public VoteComment voteUserlike(Long IdComment, String userId) {

        Comment Comment = CommentRepository.findById(IdComment).orElse(null);

        VoteComment existingVote = voteRepository.findByIdCommentAndUserId(IdComment,userId);
        VoteComment vote=new VoteComment();

        if (existingVote == null){
            vote.setUserId(userId);
            vote.setIdComment(IdComment);
            vote.setTypeVote(TypeVote.LIKE);

            if (Comment != null) {
                Comment.setLikes(Comment.getLikes() + 1);
                CommentRepository.save(Comment);
            }

            return voteRepository.save(vote);
        }
        else if (existingVote.getTypeVote()==TypeVote.DISLIKE) {
            existingVote.setTypeVote(TypeVote.LIKE);

            if (Comment != null) {
                Comment.setLikes(Comment.getLikes() + 1);
                Comment.setDislikes(Comment.getDislikes() - 1);
                CommentRepository.save(Comment);
            }

            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.LIKE) {
            existingVote.setTypeVote(TypeVote.NOVOTE);

            if (Comment != null) {
                Comment.setLikes(Comment.getLikes() - 1);
                CommentRepository.save(Comment);
            }

            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.NOVOTE) {
            existingVote.setTypeVote(TypeVote.LIKE);

            if (Comment != null) {
                Comment.setLikes(Comment.getLikes() + 1);
                CommentRepository.save(Comment);
            }

            return voteRepository.save(existingVote);
        }

        return null;
    }

    @Override
    public VoteComment voteUserdislike(Long IdComment, String userId) {
        Comment Comment = CommentRepository.findById(IdComment).orElse(null);
        VoteComment existingVote = voteRepository.findByIdCommentAndUserId(IdComment,userId);
        VoteComment vote=new VoteComment();
        if (existingVote == null){
            vote.setUserId(userId);
            vote.setIdComment(IdComment);
            vote.setTypeVote(TypeVote.DISLIKE);
            if (Comment != null) {
                Comment.setDislikes(Comment.getDislikes() + 1);
                CommentRepository.save(Comment);
            }
            return voteRepository.save(vote);
        }
        else if (existingVote.getTypeVote()==TypeVote.LIKE) {
            existingVote.setTypeVote(TypeVote.DISLIKE);
            if (Comment != null) {
                Comment.setDislikes(Comment.getDislikes() + 1);
                Comment.setLikes(Comment.getLikes() - 1);
                CommentRepository.save(Comment);
            }
            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.DISLIKE) {
            existingVote.setTypeVote(TypeVote.NOVOTE);
            if (Comment != null) {
                Comment.setDislikes(Comment.getDislikes() - 1);
                CommentRepository.save(Comment);
            }
            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.NOVOTE) {
            existingVote.setTypeVote(TypeVote.DISLIKE);
            if (Comment != null) {
                Comment.setDislikes(Comment.getDislikes() + 1);
                CommentRepository.save(Comment);
            }
            return voteRepository.save(existingVote);
        }

        return null;
    }

    @Override
    public String getUserVoteStatus(Long CommentId, String userId) {
        VoteComment existingVote = voteRepository.findByIdCommentAndUserId(CommentId, userId);

        if (existingVote == null) {
            return "NOVOTE";
        } else {
            return existingVote.getTypeVote().toString();
        }
    }

}
