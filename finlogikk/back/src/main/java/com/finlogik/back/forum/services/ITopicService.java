package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.Topic;
import com.finlogik.back.forum.entities.TopicDTO;

import java.util.List;

public interface ITopicService {
    Topic addTopic (Topic topic, String userId);
    void deleteTopic(Long topicId);
    //List<Topic> getAllTopics();
    List<TopicDTO> getAllTopics();
    //Topic getTopic(Long topicId);
    TopicDTO getTopic(Long topicId);

    Long countPostsByIdTopic(Long idTopic);


}
