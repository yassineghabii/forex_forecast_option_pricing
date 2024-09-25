package com.finlogik.back.forum.controllers;


import com.finlogik.back.forum.entities.Post;
import com.finlogik.back.forum.entities.PostDTO;
import com.finlogik.back.forum.services.IPostService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/post")
public class PostController {

    @Autowired
    IPostService postService;

    @PostMapping("/addPost/{userId}/{idTopic}")
    @ResponseBody
    public Post addPost(@RequestBody Post post,
                        @PathVariable("userId") String userId,
                        @PathVariable("idTopic") Long idTopic){

        return postService.addPost(post, userId, idTopic);
    }

    @PutMapping("/updatePost/{postId}")
    @ResponseBody
    public Post updatePost(@PathVariable Long postId,
                               @RequestBody Post post) {
        post.setIdPost(postId);
        return postService.updatePost(post);
    }

    @DeleteMapping("/deletePost/{postId}")
    @ResponseBody
    public void deletePost(@PathVariable("postId") Long postId) {

        postService.deletePost(postId);
    }

    @GetMapping("/getAllPosts")
    @ResponseBody
    public List<PostDTO> getAllPosts() {
        List<PostDTO> listPosts = postService.getAllPosts();
        return listPosts;
    }

    @GetMapping("/getPost/{postId}")
    @ResponseBody
    public Post getPost(@PathVariable("postId") Long postId) {

        return postService.getPost(postId);
    }



    @GetMapping("/byTopic/{topicId}")
    @ResponseBody
    public List<PostDTO> getPostsByTopicId(@PathVariable Long topicId) {
        List<PostDTO> listPosts = postService.getPostsByTopicId(topicId);
        return listPosts;
    }

    @GetMapping("/commentCount/{idPost}")
    public Long countCommentsByIdPost(@PathVariable Long idPost) {
        return postService.countCommentsByIdPost(idPost);
    }


}
