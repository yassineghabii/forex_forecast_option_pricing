package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.Topic;
import com.finlogik.back.forum.entities.TypeVote;
import com.finlogik.back.forum.entities.VoteTopic;
import com.finlogik.back.forum.repositories.TopicRepository;
import com.finlogik.back.forum.repositories.VoteTopicRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class VoteTopicServiceImpl implements VoteTopicService{

    private VoteTopicRepository voteRepository;
    private TopicRepository topicRepository;
    @Override
    public VoteTopic voteUserlike(Long IdTopic, String userId) {

        Topic topic = topicRepository.findById(IdTopic).orElse(null);

        VoteTopic existingVote = voteRepository.findByIdTopicAndUserId(IdTopic,userId);
        VoteTopic vote=new VoteTopic();

        if (existingVote == null){
            vote.setUserId(userId);
            vote.setIdTopic(IdTopic);
            vote.setTypeVote(TypeVote.LIKE);

            if (topic != null) {
                topic.setLikes(topic.getLikes() + 1);
                topicRepository.save(topic);
            }

            return voteRepository.save(vote);
        }
        else if (existingVote.getTypeVote()==TypeVote.DISLIKE) {
            existingVote.setTypeVote(TypeVote.LIKE);

            if (topic != null) {
                topic.setLikes(topic.getLikes() + 1);
                topic.setDislikes(topic.getDislikes() - 1);
                topicRepository.save(topic);
            }

            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.LIKE) {
            existingVote.setTypeVote(TypeVote.NOVOTE);

            if (topic != null) {
                topic.setLikes(topic.getLikes() - 1);
                topicRepository.save(topic);
            }

            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.NOVOTE) {
            existingVote.setTypeVote(TypeVote.LIKE);

            if (topic != null) {
                topic.setLikes(topic.getLikes() + 1);
                topicRepository.save(topic);
            }

            return voteRepository.save(existingVote);
        }

        return null;
    }

    @Override
    public VoteTopic voteUserdislike(Long IdTopic, String userId) {
        Topic topic = topicRepository.findById(IdTopic).orElse(null);
        VoteTopic existingVote = voteRepository.findByIdTopicAndUserId(IdTopic,userId);
        VoteTopic vote=new VoteTopic();
        if (existingVote == null){
            vote.setUserId(userId);
            vote.setIdTopic(IdTopic);
            vote.setTypeVote(TypeVote.DISLIKE);
            if (topic != null) {
                topic.setDislikes(topic.getDislikes() + 1);
                topicRepository.save(topic);
            }
            return voteRepository.save(vote);
        }
        else if (existingVote.getTypeVote()==TypeVote.LIKE) {
            existingVote.setTypeVote(TypeVote.DISLIKE);
            if (topic != null) {
                topic.setDislikes(topic.getDislikes() + 1);
                topic.setLikes(topic.getLikes() - 1);
                topicRepository.save(topic);
            }
            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.DISLIKE) {
            existingVote.setTypeVote(TypeVote.NOVOTE);
            if (topic != null) {
                topic.setDislikes(topic.getDislikes() - 1);
                topicRepository.save(topic);
            }
            return voteRepository.save(existingVote);
        }
        else if (existingVote.getTypeVote()==TypeVote.NOVOTE) {
            existingVote.setTypeVote(TypeVote.DISLIKE);
            if (topic != null) {
                topic.setDislikes(topic.getDislikes() + 1);
                topicRepository.save(topic);
            }
            return voteRepository.save(existingVote);
        }

        return null;
    }

    @Override
    public String getUserVoteStatus(Long topicId, String userId) {
        VoteTopic existingVote = voteRepository.findByIdTopicAndUserId(topicId, userId);

        if (existingVote == null) {
            return "NOVOTE";
        } else {
            return existingVote.getTypeVote().toString();
        }
    }

}
