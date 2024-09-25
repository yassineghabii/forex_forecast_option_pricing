package com.finlogik.back.forum.controllers;

import com.finlogik.back.forum.entities.Topic;
import com.finlogik.back.forum.entities.TopicDTO;
import com.finlogik.back.forum.services.ITopicService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/topic")
public class TopicController {

    @Autowired
    ITopicService topicService;

    @PostMapping("/addTopic/{userId}")
    @ResponseBody
    public Topic addTopic(@RequestBody Topic topic,
                          @PathVariable("userId") String userId){

        return topicService.addTopic(topic, userId);
    }

    @DeleteMapping("/deleteTopic/{topicId}")
    @ResponseBody
    public void deleteTopic(@PathVariable("topicId") Long topicId) {

        topicService.deleteTopic(topicId);
    }

    @GetMapping("/getAllTopics")
    @ResponseBody
    public List<TopicDTO> getAllTopics() {
        List<TopicDTO> listTopics = topicService.getAllTopics();
        return listTopics;
    }

    @GetMapping("/getTopic/{topicId}")
    @ResponseBody
    public TopicDTO getTopic(@PathVariable("topicId") Long topicId) {
        return topicService.getTopic(topicId);
    }


    @GetMapping("/postCount/{idTopic}")
    public Long countPostsByIdTopic(@PathVariable Long idTopic) {
        return topicService.countPostsByIdTopic(idTopic);
    }

}
