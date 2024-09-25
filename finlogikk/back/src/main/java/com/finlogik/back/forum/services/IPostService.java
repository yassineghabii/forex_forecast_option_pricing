package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.Post;
import com.finlogik.back.forum.entities.PostDTO;

import java.util.List;

public interface IPostService {
    Post addPost (Post post, String userId, Long idTopic);
    Post updatePost (Post post);
    void deletePost(Long postId);
    //List<Post> getAllPosts();
    List<PostDTO> getAllPosts();

    Post getPost(Long postId);

    //List<Post> getPostsByTopicId(Long topicId);
    List<PostDTO> getPostsByTopicId(Long topicId);
    Long countCommentsByIdPost(Long idPost);
}
