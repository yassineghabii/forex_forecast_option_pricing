package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.Post;
import com.finlogik.back.forum.entities.TypeVote;
import com.finlogik.back.forum.entities.VotePost;
import com.finlogik.back.forum.repositories.PostRepository;
import com.finlogik.back.forum.repositories.VotePostRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class VotePostServiceImpl implements VotePostService{

    private VotePostRepository voteRepository;
    private PostRepository PostRepository;
    @Override
    public VotePost voteUserlike(Long IdPost, String userId) {

        Post Post = PostRepository.findById(IdPost).orElse(null);

        VotePost existingVote = voteRepository.findByIdPostAndUserId(IdPost,userId);
        VotePost vote=new VotePost();

        if (existingVote == null){
            vote.setUserId(userId);
            vote.setIdPost(IdPost);
            vote.setTypeVote(TypeVote.LIKE);

            if (Post != null) {
                Post.setLikes(Post.getLikes() + 1);
                PostRepository.save(Post);
            }

            return voteRepository.save(vote);
        }
        else if (existingVote.getTypeVote()==TypeVote.DISLIKE) {
            existingVote.setTypeVote(TypeVote.LIKE);

            if (Post != null) {
                Post.setLikes(Post.getLikes() + 1);
                Post.setDislikes(Post.getDislikes() - 1);
                PostRepository.save(Post);
            }

            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.LIKE) {
            existingVote.setTypeVote(TypeVote.NOVOTE);

            if (Post != null) {
                Post.setLikes(Post.getLikes() - 1);
                PostRepository.save(Post);
            }

            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.NOVOTE) {
            existingVote.setTypeVote(TypeVote.LIKE);

            if (Post != null) {
                Post.setLikes(Post.getLikes() + 1);
                PostRepository.save(Post);
            }

            return voteRepository.save(existingVote);
        }

        return null;
    }

    @Override
    public VotePost voteUserdislike(Long IdPost, String userId) {
        Post Post = PostRepository.findById(IdPost).orElse(null);
        VotePost existingVote = voteRepository.findByIdPostAndUserId(IdPost,userId);
        VotePost vote=new VotePost();
        if (existingVote == null){
            vote.setUserId(userId);
            vote.setIdPost(IdPost);
            vote.setTypeVote(TypeVote.DISLIKE);
            if (Post != null) {
                Post.setDislikes(Post.getDislikes() + 1);
                PostRepository.save(Post);
            }
            return voteRepository.save(vote);
        }
        else if (existingVote.getTypeVote()==TypeVote.LIKE) {
            existingVote.setTypeVote(TypeVote.DISLIKE);
            if (Post != null) {
                Post.setDislikes(Post.getDislikes() + 1);
                Post.setLikes(Post.getLikes() - 1);
                PostRepository.save(Post);
            }
            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.DISLIKE) {
            existingVote.setTypeVote(TypeVote.NOVOTE);
            if (Post != null) {
                Post.setDislikes(Post.getDislikes() - 1);
                PostRepository.save(Post);
            }
            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.NOVOTE) {
            existingVote.setTypeVote(TypeVote.DISLIKE);
            if (Post != null) {
                Post.setDislikes(Post.getDislikes() + 1);
                PostRepository.save(Post);
            }
            return voteRepository.save(existingVote);
        }

        return null;
    }

    @Override
    public String getUserVoteStatus(Long PostId, String userId) {
        VotePost existingVote = voteRepository.findByIdPostAndUserId(PostId, userId);

        if (existingVote == null) {
            return "NOVOTE";
        } else {
            return existingVote.getTypeVote().toString();
        }
    }

}
